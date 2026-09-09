"""Run-scoped Hy3 call budgets shared by parallel research workers and tools."""

from __future__ import annotations

import asyncio
from collections import Counter
from contextvars import ContextVar, Token
from dataclasses import dataclass, field


@dataclass
class Hy3BudgetSnapshot:
    limit: int
    used: int
    reserved: int
    exhausted: bool
    calls_by_stage: dict[str, int]


@dataclass
class _BudgetState:
    limit: int
    reserved: int
    used: int = 0
    exhausted: bool = False
    calls_by_stage: Counter[str] = field(default_factory=Counter)
    stage_reservations: Counter[str] = field(default_factory=Counter)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class Hy3CallBudgetManager:
    """Track Hy3 calls and optionally enforce a run-wide hard limit.

    A non-positive limit disables the global cap while preserving per-stage
    accounting and reservation semantics. Positive limits retain the bounded
    behavior for deployments that explicitly opt into it.
    """

    def __init__(self, limit: int, reserved: int = 1) -> None:
        self.limit = limit
        self.reserved = 0 if limit <= 0 else min(reserved, limit)
        self._states: dict[str, _BudgetState] = {}
        self._states_lock = asyncio.Lock()
        self._current_key: ContextVar[str | None] = ContextVar(
            "deep_research_assistant_hy3_budget_key",
            default=None,
        )

    async def ensure(self, key: str) -> None:
        async with self._states_lock:
            self._states.setdefault(
                key,
                _BudgetState(limit=self.limit, reserved=self.reserved),
            )

    async def acquire(self, key: str, stage: str, *, essential: bool = False) -> bool:
        await self.ensure(key)
        state = self._states[key]
        async with state.lock:
            if state.limit <= 0:
                state.used += 1
                state.calls_by_stage[stage] += 1
                return True
            stage_reserved = sum(state.stage_reservations.values())
            ceiling = (
                state.limit
                if essential
                else state.limit - state.reserved - stage_reserved
            )
            if state.used >= ceiling:
                state.exhausted = True
                return False
            state.used += 1
            state.calls_by_stage[stage] += 1
            return True

    async def reserve_stage(self, key: str, stage: str, count: int) -> int:
        """Reserve up to ``count`` future calls exclusively for one stage."""

        if count <= 0:
            return 0
        await self.ensure(key)
        state = self._states[key]
        async with state.lock:
            if state.limit <= 0:
                state.stage_reservations[stage] += count
                return count
            already_reserved = sum(state.stage_reservations.values())
            available = max(
                state.limit - state.reserved - state.used - already_reserved,
                0,
            )
            granted = min(count, available)
            if granted:
                state.stage_reservations[stage] += granted
            if granted < count:
                state.exhausted = True
            return granted

    async def acquire_reserved(self, key: str, stage: str) -> bool:
        """Consume one previously reserved call without touching other reserves."""

        await self.ensure(key)
        state = self._states[key]
        async with state.lock:
            if state.stage_reservations[stage] <= 0:
                state.exhausted = True
                return False
            if state.limit <= 0:
                state.stage_reservations[stage] -= 1
                if state.stage_reservations[stage] <= 0:
                    del state.stage_reservations[stage]
                state.used += 1
                state.calls_by_stage[stage] += 1
                return True
            # Stage-specific work may consume its own reservation, but the
            # generic terminal reserve remains protected.
            if state.used >= state.limit - state.reserved:
                state.exhausted = True
                return False
            state.stage_reservations[stage] -= 1
            if state.stage_reservations[stage] <= 0:
                del state.stage_reservations[stage]
            state.used += 1
            state.calls_by_stage[stage] += 1
            return True

    async def acquire_current(self, stage: str) -> bool:
        """Acquire against the research thread bound to the current async task."""

        key = self._current_key.get()
        if key is None:
            # Direct tool calls outside the graph remain usable and testable.
            return True
        return await self.acquire(key, stage)

    def bind(self, key: str) -> Token[str | None]:
        return self._current_key.set(key)

    def reset_binding(self, token: Token[str | None]) -> None:
        self._current_key.reset(token)

    async def snapshot(self, key: str) -> Hy3BudgetSnapshot:
        await self.ensure(key)
        state = self._states[key]
        async with state.lock:
            return Hy3BudgetSnapshot(
                limit=state.limit,
                used=state.used,
                reserved=state.reserved + sum(state.stage_reservations.values()),
                exhausted=state.exhausted,
                calls_by_stage=dict(state.calls_by_stage),
            )

    async def release(self, key: str) -> None:
        async with self._states_lock:
            self._states.pop(key, None)
