from deep_research_assistant.artifact_store import ArtifactStore


def test_artifact_store_upsert_is_idempotent(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "research.db")
    try:
        store.put_json("thread-1", "result", {"version": 1})
        store.put_json("thread-1", "result", {"version": 2})

        assert store.get_json("thread-1", "result") == {"version": 2}
    finally:
        store.close()
