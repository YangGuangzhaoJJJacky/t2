from typing import Tuple

import fishfarm
import vllm
from datasets import load_dataset
from fishfarm.models.vllm_model import VLLMModel
from fishfarm.tasks.MCQ_math import MCQMathTask, MathSample

from .base import Task, get_download_dir


class AquaRatTask(Task):
    def __init__(
        self, node=0
    ):
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
        self.system_msg = (
            "Below is an math MCQ. Calculate and Answer it. Stop calculate when the conclusion is clear. Answer briefly. "
            "Stop calculate when the conclusion is clear. Answer briefly. Do not include unnecessary steps."
            "Return the final answer you select in A,B,C,D,E directly at last."
        )

        self.target_metric_train = "acc"
        self.target_metric_valid = self.target_metric_train
        self.target_metric_test = self.target_metric_train
        self.target_metric_transfer = self.target_metric_train
        self.has_transfer_split = False
        self.has_training_split = True
        self.node = node

    def get_train_data(self):
        print(f"#############current node {self.node} ###########")
        train_data = load_dataset("yangguangzhaojjj/aqua_rat", split=f"subset_{self.node}")
        train_data = train_data.select(range(5000))
        train_size = len(train_data)
        train_ix = range(0, train_size-256)
        valid_ix = range(train_size-256, train_size)
        return train_data, train_ix, valid_ix

    def get_rewards(self, res):
        rewards = [1.0 if x["correct"] else -1.0 for x in res.sample_details]
        return rewards

    def get_evaluator(self) -> Tuple:
        res = []
        dataset_list = [load_dataset("yangguangzhaojjj/aqua_rat", split=f"subset_{self.node}"),
                        load_dataset("deepmind/aqua_rat", "raw", split="test")]
        for dataset in dataset_list:
            samples = []
            for sample in dataset:
                answer = sample["correct"]
                answer = str(answer) if answer is not None else None
                options_str = "\n".join(sample["options"])
                samples.append(
                    MathSample(
                        problem=sample["question"]+"\nOptions: \n"+options_str,
                        answer=answer,
                    )
                )
            res.append(
                MCQMathTask(
                    samples=samples,
                    context_messages=[
                        fishfarm.Message("system", self.system_msg),
                    ],
                    languages=[],
                )
            )
            #print("res",res)
        return tuple(res)

    def get_prompt(self, tokenizer, samples, ix, model_id):
        chat_template = self.model_to_template[model_id]
        context_msg = {"role": "system", "content": self.system_msg}
        options_str = "\n".join(samples["options"][ix])
        user_msg = {"role": "user", "content": samples["question"][ix]+"\nOptions: \n"+options_str}
        prompt = tokenizer.apply_chat_template(
            conversation=[context_msg, user_msg],
            chat_template=chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )
        #print("prompt",prompt)
        return prompt

    def get_vllm_model(self, model_id) -> VLLMModel:
        """Load a vLLM model."""
        model = vllm.LLM(
            model_id,
            max_model_len=1500,
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
                max_tokens=1300,
                stop=["Instruction:", "Instruction", "Response:", "Response"],
                repetition_penalty=1.0,
            ),
            chat_template=chat_template,
        )
        return vllm_model