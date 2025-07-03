import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence
import numpy as np
import huggingface_hub
from jiwer import cer
import re

from ..imports import try_import
from ..models import GenerationRequest, Message, Model
from .base import Task, TaskResult

with try_import() as _imports:
    #import fasttext
    pass

_imports.check()

def normalize_text(text: str) -> str:
    text = re.sub(r'[^\u3040-\u30FF\u4E00-\u9FFF]', '', text)
    return text

@dataclass
class STTSample:

    audio: np.ndarray
    sr: int
    transcription: str


def mean(iterable: Iterable[float]) -> float:
    total, count = 0.0, 0
    for x in iterable:
        total += x
        count += 1
    return total / count

class Audio_llm_Task(Task):
    def __init__(
        self,
        samples: Sequence[STTSample],
    ):
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
        samples = [self.samples[sample_id] for sample_id in sample_ids]

        sample_details = []
        for sample, result in zip(samples, model.generate_with_embedding(samples)):
            output = result.generation
            prediction = output
            
            sample_details.append(
                dict(
                    problem=sample.audio,
                    output=output,
                    answer=sample.transcription,
                    prediction=prediction,
                    correct = cer(normalize_text(prediction), normalize_text(sample.transcription)),
                    prompt_embeds = result.embedding_prompt
                )
            )
        print(
            sample_details[0]["output"],
            sample_details[0]["answer"],
            sample_details[0]["correct"]
        )

        aggregate_metrics = {"cer": mean(sd["correct"] for sd in sample_details)}

        return TaskResult(
            aggregate_metrics=aggregate_metrics, sample_details=sample_details
        )
