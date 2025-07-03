import torch
import os
import librosa
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, WhisperModel, WhisperFeatureExtractor
import torch.nn as nn
import vllm
from typing import Union, Optional

os.environ["HF_HOME"] = "/mnt/data-raid/yangguangzhao/.cache"

class Projector(nn.Module):
    """ A projector module to transform speech embeddings to LLM embedding space. """

    def __init__(self, speech_encoder_hidden_size, llm_hidden_size):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(speech_encoder_hidden_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, llm_hidden_size),
        )

    def forward(self, x):
        return self.proj(x)


class AudioToEmbedding(torch.nn.Module):

    def __init__(self, encoder_id="openai/whisper-large-v3", llm_model_name="models/Qwen3-0.6B"):
        super().__init__()
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(encoder_id)
        self.speech_encoder = WhisperModel.from_pretrained(encoder_id).encoder
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)
        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name, device_map="cuda:0", trust_remote_code=True)
        
        self.downsample_factor = 5
        self.projector = Projector(
            self.speech_encoder.config.hidden_size * self.downsample_factor,
            self.llm_model.config.hidden_size,
        )
        
        # Freeze the parameters of the pretrained models
        for param in self.speech_encoder.parameters():
            param.requires_grad = False
        for param in self.llm_model.parameters():
            param.requires_grad = False
        
        self.init_prompts()

    def init_prompts(self):
        """ Initialize embeddings for static prompts used with the LLM. """
        user_prompt_text = "<|im_start|>system\n"
        assistant_prompt_text = "<|im_end|>\n<|im_start|>user\n音声認識ください<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        self.user_prompt_embeds, self.user_mask = self.get_text_embedding(
            user_prompt_text, return_attention_mask=True)
        self.assistant_prompt_embeds, self.assistant_mask = self.get_text_embedding(
            assistant_prompt_text, return_attention_mask=True)

    def load_checkpoint(self, checkpoint_path="slam_asr/epoch=0-step=15625.ckpt"):
        """ Load projector weights from checkpoint. """
        print(f"Loading checkpoint from: {checkpoint_path}")
        try:
            pretrained_dict = torch.load(checkpoint_path, weights_only=False)
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict['state_dict'].items() if k in model_dict
            }
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict=False)
            print("✅ Checkpoint loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")

    def _clean_audio(self, audio: np.ndarray, sampling_rate: int = 16000):
        """ Clean and prepare audio data. """
        max_length = 16000 * 30
        audio = audio[:max_length]
        
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.tensor(audio, dtype=torch.float32)
        else:
            audio_tensor = audio.float()
        
        if audio_tensor.ndim == 2:
            audio_tensor = torch.mean(audio_tensor, dim=0)
        if sampling_rate != 16000:
            import torchaudio.transforms as T
            resampler = T.Resample(sampling_rate, 16000)
            audio_tensor = resampler(audio_tensor)
        
        return audio_tensor.numpy()

    def get_input_embeddings(self, audio: Union[str, np.ndarray], sr: int = 16000):
       
        if isinstance(audio, str):
            audio, sr = load_audio(audio)
        
        audio = self._clean_audio(audio, sr)
        
        with torch.no_grad():
            audio_features = self.feature_extractor(
                audio, 
                return_tensors="pt", 
                sampling_rate=16000,
                return_attention_mask=True
            )
            
            input_features = audio_features.input_features.to(self.speech_encoder.device)
            encoded_features = self.speech_encoder(input_features)
            downsampled_features = self.downsample(
                encoded_features.last_hidden_state, k=self.downsample_factor
            )
            projected_features = self.projector(downsampled_features)
            inputs_embeds = torch.cat([
                self.user_prompt_embeds.to(projected_features.device, dtype=projected_features.dtype),
                projected_features,
                self.assistant_prompt_embeds.to(projected_features.device, dtype=projected_features.dtype)
            ], dim=1)
        
            return inputs_embeds.squeeze(0)

    @staticmethod
    def downsample(features, k):
        """ Downsample the feature dimension by a factor of k. """
        batch_size, seq_len, hidden_size = features.shape
        if seq_len % k != 0:
            seq_len = (seq_len // k) * k
            features = features[:, :seq_len, :]
        
        downsampled_features = features.view(
            batch_size, seq_len // k, k * hidden_size)
        return downsampled_features

    def get_text_embedding(self, text, return_attention_mask=False):
        """ Generate embeddings for given text using the tokenizer and LLM. """
        tokens = self.llm_tokenizer(
            text, return_tensors="pt", padding=False, truncation=True, max_length=1024
        )
        token_ids = tokens.input_ids.to(self.llm_model.device)
        attention_mask = tokens.attention_mask.to(self.llm_model.device) if return_attention_mask else None
        
        embedding_layer = self.llm_model.get_input_embeddings()
        embeddings = embedding_layer(token_ids)
        
        if return_attention_mask:
            return embeddings, attention_mask
        else:
            return embeddings


def load_audio(audio_path, target_sr=16000):
    """ Load and resample an entire audio file. """
    audio, sr = librosa.load(audio_path, sr=None)  # Load the full audio
    audio = librosa.to_mono(audio)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio, target_sr


def segment_audio(audio, segment_length=30, sr=16000):
    """ Segment audio into chunks of a specified length in seconds. """
    segment_samples = segment_length * sr
    return [audio[i:i+segment_samples] for i in range(0, len(audio), segment_samples)]


def transcribe_audio(audio_to_embedding_model, audio_path, llm_model_name="models/Qwen3-0.6B", device='cuda'):
    
    audio, sr = load_audio(audio_path)
    segments = segment_audio(audio, segment_length=20, sr=sr)
    llm = vllm.LLM(
        model=llm_model_name,
        gpu_memory_utilization=0.6,
        enforce_eager=True,
        dtype="float16",
        trust_remote_code=True,
        disable_custom_all_reduce=True,
        enable_prompt_embeds=True,
        max_model_len=2048
    )
    
    sampling_params = vllm.SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=256,
        stop=["<|im_end|>", "</s>"]
    )
    
    full_transcription = []
    
    for i, segment in enumerate(segments):
        if len(segment) > 0:
            print(f"处理音频段 {i+1}/{len(segments)}...")
            
            inputs_embeds = audio_to_embedding_model.get_input_embeddings(segment, sr=sr)
            
            input_embeds_dict = [{"prompt_embeds": inputs_embeds} for _ in range(2)]
            
            outputs = llm.generate(input_embeds_dict, sampling_params)
            transcription = outputs[0].outputs[0].text
            full_transcription.append(transcription)
            
            print(f"段 {i+1} 转录结果: {transcription}")
    
    return ' '.join(full_transcription)


# Main execution block
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = 'slam_asr/epoch=0-step=15625.ckpt'
    audio_path = 'slam_asr/example.wav'


    audio_to_embedding_model = AudioToEmbedding()
    audio_to_embedding_model.load_checkpoint(checkpoint_path)
    audio_to_embedding_model.eval()
    transcription = transcribe_audio(
        audio_to_embedding_model, 
        audio_path, 
        llm_model_name="models/Qwen3-0.6B",
        device=device
    )
    
    print("转录结果:", transcription)