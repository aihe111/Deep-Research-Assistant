import asyncio

from deep_research_assistant.hy3_budget import Hy3CallBudgetManager


def test_stage_reservation_cannot_be_consumed_by_ordinary_calls() -> None:
    async def scenario() -> None:
        manager = Hy3CallBudgetManager(limit=6, reserved=1)
        key = "reserved-compression"

        assert await manager.acquire(key, "researcher")
        assert await manager.acquire(key, "researcher")
        assert await manager.reserve_stage(key, "research_compression", 2) == 2

        # One ordinary slot remains after preserving the final-report and two
        # compression reservations.
        assert await manager.acquire(key, "tavily_webpage_summary")
        assert not await manager.acquire(key, "tavily_webpage_summary")

        assert await manager.acquire_reserved(key, "research_compression")
        assert await manager.acquire_reserved(key, "research_compression")
        assert await manager.acquire(key, "final_report", essential=True)

        snapshot = await manager.snapshot(key)
        assert snapshot.used == 6
        assert snapshot.calls_by_stage == {
            "researcher": 2,
            "tavily_webpage_summary": 1,
            "research_compression": 2,
            "final_report": 1,
        }

    asyncio.run(scenario())


def test_zero_limit_tracks_calls_without_enforcing_a_global_cap() -> None:
    async def scenario() -> None:
        manager = Hy3CallBudgetManager(limit=0, reserved=2)
        key = "unlimited-research"

        for _ in range(200):
            assert await manager.acquire(key, "researcher")
        assert await manager.reserve_stage(key, "research_compression", 3) == 3
        for _ in range(3):
            assert await manager.acquire_reserved(key, "research_compression")
        assert await manager.acquire(key, "final_report", essential=True)

        snapshot = await manager.snapshot(key)
        assert snapshot.limit == 0
        assert snapshot.used == 204
        assert snapshot.reserved == 0
        assert snapshot.exhausted is False
        assert snapshot.calls_by_stage == {
            "researcher": 200,
            "research_compression": 3,
            "final_report": 1,
        }

    asyncio.run(scenario())
