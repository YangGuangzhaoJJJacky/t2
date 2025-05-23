from dataclasses import dataclass
import json
from typing import Sequence, Optional
from .base import Task, TaskResult
from ..models import GenerationRequest, Message, Model

@dataclass
class IotSample:
    prompt: str
    user_input: str
    ai_output: str

class IotCommandMatchTask(Task):
    def __init__(self, samples: Sequence):
        self.samples = list(samples)

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def evaluate(
        self,
        model: Model,
        sample_ids: Optional[Sequence[int]] = None,
    ) -> TaskResult:
        if sample_ids is None:
            sample_ids = range(len(self.samples))
        samples = [self.samples[i] for i in sample_ids]

        requests = []
        for sample in samples:
            messages = [Message(role="system", content=sample.prompt)]
            messages.append(Message(role="user", content=sample.user_input))
            requests.append(GenerationRequest(messages=messages))

        sample_details = []
        for sample, result in zip(samples, model.generate(requests)):
            output = result.generation

            try:
                pred_json = json.loads(output)
                prediction = pred_json.get("command", None)
            except Exception:
                prediction = None

            try:
                gt_json = json.loads(sample.ai_output)
                ground_truth = gt_json.get("command", None)
            except Exception:
                ground_truth = None

            correct = (prediction is not None) and (prediction == ground_truth)

            sample_details.append(
                dict(
                    prompt=sample.prompt,
                    user_input=sample.user_input,
                    output=output,
                    prediction=prediction,
                    answer=ground_truth,
                    correct=correct,
                )
            )

        aggregate_metrics = {
            "acc": sum(sd["correct"] for sd in sample_details) / len(sample_details)
        }

        return TaskResult(
            aggregate_metrics=aggregate_metrics,
            sample_details=sample_details
        )
