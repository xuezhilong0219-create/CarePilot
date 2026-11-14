from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import Any

from .ledger import canonical_json


class BudgetExceeded(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class RunBudget:
    max_steps: int = 20
    max_tokens: int = 50_000
    max_seconds: float = 120.0

    def __post_init__(self) -> None:
        if self.max_steps < 0 or self.max_tokens < 0 or self.max_seconds < 0:
            raise ValueError("budgets must be nonnegative")


class RunGuard:
    def __init__(self, budget: RunBudget, *, clock: Callable[[], float] = time.monotonic):
        self.budget = budget
        self.clock = clock
        self.started_at = clock()
        self.steps = 0
        self.tokens = 0
        self._last: tuple[str, str] | None = None
        self._consecutive = 0
        self._counts: Counter[tuple[str, str]] = Counter()
        self._reason: str | None = None

    def step(self, kind: str, *, tool: str | None = None, args: dict[str, Any] | None = None, tokens: int = 0) -> None:
        if self._reason:
            raise BudgetExceeded(self._reason)
        if not isinstance(tokens, int) or tokens < 0:
            raise ValueError("tokens must be a nonnegative integer")
        key = (tool, canonical_json(args or {})) if tool is not None else None
        self.steps += 1
        self.tokens += tokens
        if key is not None:
            self._counts[key] += 1
            self._consecutive = self._consecutive + 1 if key == self._last else 1
        else:
            self._consecutive = 0
        self._last = key
        if self.steps > self.budget.max_steps:
            self._reason = "steps"
        elif self.tokens > self.budget.max_tokens:
            self._reason = "tokens"
        elif self.clock() - self.started_at > self.budget.max_seconds:
            self._reason = "time"
        elif key is not None and (self._consecutive >= 3 or self._counts[key] >= 4):
            self._reason = "loop"
        if self._reason:
            raise BudgetExceeded(self._reason)
