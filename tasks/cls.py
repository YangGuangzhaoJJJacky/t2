import re
from dataclasses import dataclass
from typing import Iterable, Tuple

import datasets
import fishfarm
import vllm
from fishfarm.models.vllm_model import VLLMModel
from fishfarm.tasks.base import TaskResult
from datasets import load_dataset

from .base import Task, get_download_dir


def mean(iterable: Iterable[float]) -> float:
    total, count = 0.0, 0
    for x in iterable:
        total += x
        count += 1
    return total / count


def extract_ans(text):
    """Fetch the string within \\boxed{}."""
    match = re.search(r"\\boxed{([^}]*)}", text)
    if match:
        return match.group(1)  # Return the content inside the \boxed{}
    else:
        return None  # Return None if no match is found


@dataclass
class CategorySample:
    question: str
    label: str


class CategoryClassificationTask(fishfarm.tasks.base.Task):
    def __init__(
        self,
        samples,
        context_messages,
    ):
        self.samples = list(samples)
        self.context_messages = context_messages

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def evaluate(
        self,
        model,
        sample_ids,
    ):
        if sample_ids is None:
            sample_ids = range(len(self.samples))
        samples = [self.samples[sample_id] for sample_id in sample_ids]

        requests = []
        for sample in samples:
            messages = list(self.context_messages)
            messages.append(fishfarm.Message(role="user", content=sample.question))
            requests.append(fishfarm.models.GenerationRequest(messages=messages))

        sample_details = []
        for sample, result in zip(samples, model.generate(requests)):
            output = result.generation
            prediction = extract_ans(output)

            sample_details.append(
                dict(
                    question=sample.question,
                    label=sample.label,
                    output=output,
                    prediction=prediction,
                    correct=sample.label == prediction,
                )
            )

        # 打印调试信息 - 显示第一个样本的详细结果
        if sample_details:
            first_sample = sample_details[0]
            print(f"question  : {first_sample['question'][:200]}...")
            print(f"label  : {first_sample['label']}")
            print(f"output  : {first_sample['output']}")
            print(f"prediction  : {first_sample['prediction']}")
            print(f"correct  : {first_sample['correct']}")
            

        aggregate_metrics = {
            "acc": mean(
                float(sd["correct"]) if isinstance(sd["correct"], (bool)) else 0.0
                for sd in sample_details
            )
        }
        return TaskResult(
            aggregate_metrics=aggregate_metrics, sample_details=sample_details
        )


class ClsTask(Task):
    def __init__(self, node=0):
        
        self.node = node
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
        with open("my_utils/cls.txt", "r") as f:
            self.system_msg = f.read()

        self.system_msg += """
    # Analyze the given question and classify it into one of 10 categories as I mentioned above. 

    Instructions:
    - If a question spans multiple categories, choose the most dominant one.
    - DONOT calculate the question, just classify it.
    - Provide your final classification NUMBER within \\boxed{} notation. Example: \\boxed{1}

    Format your response as follows:
    Classification: \\boxed{category}
    """
        self.target_metric_train = "acc"
        self.target_metric_valid = self.target_metric_train
        self.target_metric_test = self.target_metric_train
        self.target_metric_transfer = self.target_metric_train
        self.has_transfer_split = False
        self.has_training_split = True
        self.num_samples_per_task = 400  # Hard code 400 samples per task
        self.task_datasets = [
            # load_dataset("yangguangzhaojjj/aqua_rat_cls", split=f"cls_{self.node+1}").select(range(1000)),  # train dataset
            load_dataset("yangguangzhaojjj/aqua_rat_random", split=f"subset_{self.node}"),
            load_dataset("yangguangzhaojjj/aqua_rat_test", split="test")  # test dataset
        ]
        
        # 加载数据并转换为CategorySample格式
        self.train_samples, self.test_samples = self._load_and_prepare_data()

    def _load_and_prepare_data(self):
        """加载数据集并转换为CategorySample格式"""
        train_dataset = self.task_datasets[0]
        test_dataset = self.task_datasets[1]
        
        # 转换训练数据
        train_samples = []
        for item in train_dataset:
            question = item['question'] + "\nOptions:\n" + "\n".join(item['options'])
            label = str(item['cls'])  # 确保label是字符串格式
            train_samples.append(CategorySample(question=question, label=label))
        
        # 转换测试数据
        test_samples = []
        for item in test_dataset:
            question = item['question'] + "\nOptions:\n" + "\n".join(item['options'])
            label = str(item['cls'])  # 确保label是字符串格式
            test_samples.append(CategorySample(question=question, label=label))
        
        return train_samples, test_samples

    def get_train_data(self, num_samples=400):
        print(f"#############current node {self.node} ###########")
        train_size = len(self.train_samples)
        train_ix = range(0, train_size-256)
        valid_ix = range(train_size-256, train_size)

        return self.train_samples, train_ix, valid_ix

    def get_rewards(self, res):
        rewards = [1.0 if x["correct"] else -1.0 for x in res.sample_details]
        return rewards

    def get_evaluator(self) -> Tuple:
        # Build cls dataset here with training tasks.
        res = []
        for samples in [self.train_samples, self.test_samples]:
            res.append(
                CategoryClassificationTask(
                    samples=samples,
                    context_messages=[
                        fishfarm.Message("system", self.system_msg),
                    ],
                )
            )

        return tuple(res)

    def get_prompt(self, tokenizer, samples, ix, model_id):
        chat_template = self.model_to_template[model_id]
        context_msg = {"role": "system", "content": self.system_msg}
        user_msg = {"role": "user", "content": samples[ix].question}
        prompt = tokenizer.apply_chat_template(
            conversation=[context_msg, user_msg],
            chat_template=chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt

    def get_vllm_model(self, model_id) -> VLLMModel:
        """Load a vLLM model."""
        model = vllm.LLM(
            model_id,
            max_model_len=2048,
            gpu_memory_utilization=0.6,
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
            audio_to_embedding_model=None,
        )
        return vllm_model
