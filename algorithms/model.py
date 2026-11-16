from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class LearningModule:
    name: str
    time: int
    score: int


@dataclass
class Result:
    algorithm: str
    value: int
    indices: List[int] = field(default_factory=list)
    operations: int = 0


def validate_instance(modules, budget):
    if budget < 0:
        raise ValueError("negative budget")
    for m in modules:
        if m.time < 1 or m.score < 0:
            raise ValueError(f"invalid module: {m}")


def evaluate(modules, indices):
    total_time = sum(modules[i].time for i in indices)
    total_score = sum(modules[i].score for i in indices)
    return total_time, total_score
