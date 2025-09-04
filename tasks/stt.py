from typing import Tuple

import fishfarm
import vllm
from datasets import load_dataset
from huggingface_hub import login
import os
import numpy as np
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from fishfarm.models.vllm_model import VLLMModel
from fishfarm.tasks.voice import Audio_llm_Task, STTSample
from slam_asr.inference import AudioToEmbedding

from .base import Task, get_download_dir

load_dotenv(".env") 
login(os.environ["HF_TOKEN"])
from huggingface_hub import whoami
print("✅ Logged in as:", whoami()["name"])

class STTTask(Task):
    def __init__(
        self, node=0
    ):  
        self.target_metric_train = "cer"
        self.target_metric_valid = self.target_metric_train
        self.target_metric_test = self.target_metric_train
        self.target_metric_transfer = self.target_metric_train
        self.has_transfer_split = False
        self.has_training_split = True
        self.node = node
         # build audio_to_embedding_model
        self.audio_to_embedding_model = AudioToEmbedding()
        self.audio_to_embedding_model.load_checkpoint()
        self.audio_to_embedding_model.eval()

    def get_train_data(self):
        print(f"#############current node {self.node} ###########")
        # train_data = load_dataset("RecoseleInc/TTS-jp", "filtered_dataset_14-clean", split="train")
        # train_data = load_dataset("japanese-asr/ja_asr.jsut_basic5000", split="test")
        train_data = load_dataset("yangguangzhaojjj/travel1000",  split="train")
        train_data = train_data.select(range(200*self.node, 200*(self.node+1)))
        train_size = len(train_data)
        train_ix = range(0, train_size-128)
        valid_ix = range(train_size-128, train_size)
        return train_data, train_ix, valid_ix

    def get_rewards(self, res):
        rewards = [max(-1.0, min(1.0, 1.0 - 2.0 * x["correct"])) for x in res.sample_details]
        return rewards

    def get_evaluator(self) -> Tuple:
        res = []
        dataset_list = [load_dataset("yangguangzhaojjj/travel1000", split="train"),
                        load_dataset("yangguangzhaojjj/travel1000", split="train").select(range(800,1000))]
        for dataset in dataset_list:
            samples = []
            for sample in dataset:
                audio_data = sample["audio"]
                transcription = sample.get("transcription") or sample.get("text") or ""
                samples.append(
                    STTSample(
                        transcription=transcription,
                        audio=np.array(audio_data["array"], dtype=np.float32),
                        sr=audio_data["sampling_rate"]
                    )
                )
            res.append(
                Audio_llm_Task(
                    samples=samples,
                )
            )
            #print("res",res)
        return tuple(res)
    
    def get_prompt(self):
        return 

    def get_vllm_model(self, model_id) -> VLLMModel:
        """Load a vLLM model."""
        #tokenizer = AutoTokenizer.from_pretrained(model_id)
        #tf_model = AutoModelForCausalLM.from_pretrained(model_id)
        model = vllm.LLM(
            model_id,
            max_model_len=400,
            gpu_memory_utilization=0.6,
            enforce_eager=True,
            dtype="float16",
            enable_prompt_embeds=True,
            download_dir=get_download_dir(),
        )
        # This may change with vLLM versions.
        m = model.llm_engine.model_executor.driver_worker.model_runner.model
        for _, param in m.named_parameters():
            param.requires_grad = False
        
        vllm_model = VLLMModel(
            model,
            sampling_params=vllm.SamplingParams(
                temperature=0,
                top_p=1,
                max_tokens=200,
                stop=["Instruction:", "Instruction", "Response:", "Response"],
                repetition_penalty=1.0,
            ),
            chat_template=None,
            audio_to_embedding_model=self.audio_to_embedding_model,
        )
        #return vllm_model, tokenizer, transformers_model
        return vllm_model