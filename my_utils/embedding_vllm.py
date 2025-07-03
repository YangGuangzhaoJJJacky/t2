import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
import os
import vllm
from typing import TypedDict, NotRequired

class EmbedsPrompt(TypedDict):
    prompt_embeds: torch.Tensor
    cache_salt: NotRequired[str]


def tokenize_text(text, tokenizer):
    return tokenizer(
        text, return_tensors="pt", padding=False, truncation=True, max_length=1024, add_special_tokens=False
    )

def get_text_embedding(text, model, tokenizer, return_attention_mask=False):
    tokens = tokenize_text(text, tokenizer)
    token_ids = tokens.input_ids.to(model.device)
    attention_mask = tokens.attention_mask.to(model.device)
    embedding_layer = model.get_input_embeddings()
    embeddings = embedding_layer(token_ids)

    if return_attention_mask:
        return embeddings.to(dtype=model.dtype), attention_mask
    else:
        return embeddings.to(dtype=model.dtype)

llm_path = "models/Qwen3-0.6B"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)
llm_tokenizer.pad_token = llm_tokenizer.eos_token
llm_model = AutoModelForCausalLM.from_pretrained(llm_path)

text = "<|im_start|>user\n<|im_end|>\n<|im_start|>user\n AIですか？<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prompt_embeds = get_text_embedding(text,llm_model,llm_tokenizer)
input_embeds: EmbedsPrompt = {
    "prompt_embeds": [prompt_embeds,prompt_embeds]
}

sampling_params = vllm.SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
    stop=["<|im_end|>", "</s>"]
)

llm = vllm.LLM(model=llm_path,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            dtype="float16",
            max_model_len=200,
            enable_prompt_embeds=True)
               
outputs = llm.generate([input_embeds],sampling_params)

print(outputs)
