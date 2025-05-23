from typing import Tuple

import fishfarm
import vllm
from datasets import load_dataset
from fishfarm.models.vllm_model import VLLMModel
from fishfarm.tasks.competation_math import (LatexFormatMathTask, MathSample,
                                             last_boxed_only_string,
                                             remove_boxed)
from fishfarm.tasks.iot_acc import IotSample,IotCommandMatchTask
from .base import Task, get_download_dir


class IotTask(Task):
    def __init__(self):
        self.model_to_template = {
            "models/Qwen3-0.6B": (
                "{% set loop_messages = messages %}"
                "{% for message in loop_messages %}"
                "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] | trim + '<|im_end|>' }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<|im_start|>assistant\\n' }}"
                "{% endif %}"
                "<think>\n\n</think>\n\n"
            )
        }
        
        self.target_metric_train = "acc"
        self.target_metric_valid = self.target_metric_train
        self.target_metric_test = self.target_metric_train
        self.target_metric_transfer = self.target_metric_train
        self.has_transfer_split = False
        self.has_training_split = True

    def get_train_data(
        self,
    ):
        train_data = load_dataset("dataset/iot", split="train")
        train_size = len(train_data)
        train_ix = range(0, train_size, 2)
        valid_ix = range(1, train_size, 2)
        return train_data, train_ix, valid_ix


    def get_rewards(self, res):
        rewards = [1.0 if x["correct"] else -1.0 for x in res.sample_details]
        return rewards

    def get_evaluator(
        self,
    ) -> Tuple:
        dataset = load_dataset("dataset/iot", split="train")
        #print(dataset)
        samples = []
        for sample in dataset:
            samples.append(
                IotSample(
                    prompt=sample["prompt"], user_input=sample["user_input"], ai_output=sample["ai_output"]
                )
            )
        #print(samples)
        train_eval = IotCommandMatchTask(samples=samples)
        test_eval = IotCommandMatchTask(samples=samples,)

        return (train_eval, test_eval)

    def get_prompt(self, tokenizer, samples, ix, model_id):
        #print(samples)
        chat_template = self.model_to_template[model_id]
        context_msg = {"role": "system", "content": samples[int(ix)]["prompt"]}
        user_msg = {"role": "user", "content": samples[int(ix)]["user_input"]}
        prompt = tokenizer.apply_chat_template(
            conversation=[context_msg, user_msg],
            chat_template=chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )
        #print(prompt)
        return prompt

    def get_vllm_model(self, model_id) -> VLLMModel:
        """Load a vLLM model."""
        model = vllm.LLM(
            model_id,
            max_model_len=2048,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            dtype="float16",
            download_dir=get_download_dir(),
        )
        chat_template = self.model_to_template[model_id]
        # This may change with vLLM versions.
        m = model.llm_engine.model_executor.driver_worker.model_runner.model
        for _, param in m.named_parameters():
            param.requires_grad = False
        vllm_model = VLLMModel(
            model,
            sampling_params=vllm.SamplingParams(
                temperature=0,
                top_p=1,
                max_tokens=1024,
                stop=["Instruction:", "Instruction", "Response:", "Response"],
                repetition_penalty=1.0,
            ),
            chat_template=chat_template,
        )
        return vllm_model
