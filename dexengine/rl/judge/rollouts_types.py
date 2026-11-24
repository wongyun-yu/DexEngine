"""Result records of the clip-balanced judge loop: per-clip totals and per-batch statistics."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClipResult:
    """Per-clip statistics accumulated across batches."""

    clip_id: int
    clip_name: str
    completed: int = 0
    early_terminated: int = 0
    total_steps: int = 0
    excl_a_batches: int = 0  # real batches that left this clip out of the PARK gates (gate-A exclusion); still scored

    @property
    def total(self) -> int:
        return self.completed + self.early_terminated

    @property
    def completion_rate(self) -> float:
        return self.completed / max(self.total, 1)


@dataclass
class BatchResult:
    """Statistics for one evaluation batch."""

    batch_idx: int
    batch_size: int
    completed: int = 0
    early_terminated: int = 0
    step_sum: int = 0
    clip_counts: dict = field(default_factory=dict)  # clip_id -> (completed, early)
    park_e: dict = field(default_factory=dict)  # (clip_id, side) -> (E samples, samples) at the cut (Park.e_rows)
    park_excl_a: dict = field(default_factory=dict)  # (clip_id, side) -> median |dz| mm, gate-A-excluded (Park.excl_a)

    @property
    def total(self) -> int:
        return self.completed + self.early_terminated

    @property
    def completion_rate(self) -> float:
        return self.completed / max(self.total, 1)

    @property
    def early_rate(self) -> float:
        return self.early_terminated / max(self.total, 1)

    @property
    def mean_ep_len(self) -> float:
        return self.step_sum / max(self.total, 1)
