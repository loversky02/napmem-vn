from napmem import MemoryPyramid, MemoryRecord, MemoryTools, Message


def test_memory_tools_navigate_between_layers(tmp_path):
    pyramid = MemoryPyramid(tmp_path)
    pyramid.append_message(Message(
        message_id="m1",
        user_id="u1",
        session_id="s1",
        role="user",
        content="I visited Tampa beach to relax after the turtle trip.",
        timestamp="2022-11-11T10:00:00",
    ))
    pyramid.add_record(MemoryRecord(
        record_id="r1",
        user_id="u1",
        record_type="event",
        content="User visited Tampa beach for peace and relaxation.",
        created_at="2022-11-11T10:00:00",
        updated_at="2022-11-11T10:00:00",
        source_message_ids=["m1"],
    ))
    pyramid.upsert_topic_track(
        "nate-travel.md",
        "# Nate Travel\n\nEvidence: r1. Tampa beach is in Florida.\n",
    )
    pyramid.update_profile("# Profile\n\nEnjoys peaceful beach trips.\n")

    tools = MemoryTools(pyramid)
    assert tools.search_conversations("Tampa turtle")[0]["message_id"] == "m1"
    assert tools.search_records("beach relaxation")[0]["record_id"] == "r1"
    assert tools.get_records(["r1"])[0]["source_message_ids"] == ["m1"]
    assert "Florida" in tools.read_file("nate-travel.md")
    assert tools.used_memory()


def test_record_provenance_is_validated(tmp_path):
    pyramid = MemoryPyramid(tmp_path)
    try:
        pyramid.add_record(MemoryRecord(
            record_id="bad",
            user_id="u1",
            record_type="fact",
            content="Unsupported fact.",
            created_at="now",
            updated_at="now",
            source_message_ids=["missing"],
        ))
    except ValueError as exc:
        assert "missing messages" in str(exc)
    else:
        raise AssertionError("expected provenance validation failure")
