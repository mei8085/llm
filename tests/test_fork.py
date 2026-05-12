import pytest
import datetime
import json
import sqlite_utils
from click.testing import CliRunner
from llm.cli import cli
from llm.migrations import migrate
from llm.utils import monotonic_ulid


def test_conversation_fork_basic(mock_model):
    mock_model.enqueue(["Hello"])
    mock_model.enqueue(["World"])
    mock_model.enqueue(["Forked"])

    original = mock_model.conversation()
    r1 = original.prompt("first prompt")
    r1.text()

    r2 = original.prompt("second prompt")
    r2.text()

    assert len(original.responses) >= 2
    assert original.responses[-2].text() == "Hello"
    assert original.responses[-1].text() == "World"

    # Fork from the first response (not the last one)
    first_response = original.responses[0]
    forked = original.fork(first_response)

    assert forked.id != original.id
    assert len(forked.responses) == 1
    assert forked.responses[0] == first_response
    assert forked.parent_response_id == first_response.id

    r3 = forked.prompt("forked prompt")
    r3.text()

    assert r3.parent_response_id == first_response.id


def test_conversation_fork_not_in_conversation(mock_model):
    mock_model.enqueue(["Hello"])

    original = mock_model.conversation()
    r1 = original.prompt("first prompt")
    r1.text()

    mock_model.enqueue(["Other"])
    other_conv = mock_model.conversation()
    other_r = other_conv.prompt("other prompt")
    other_r.text()

    with pytest.raises(ValueError, match="Response is not part of this conversation"):
        original.fork(other_r)


def test_fork_log_to_db(user_path):
    log_path = str(user_path / "logs_fork.db")
    db = sqlite_utils.Database(log_path)
    migrate(db)

    model = __import__("llm").get_model("echo")
    conversation = model.conversation()

    r1 = conversation.prompt("hello")
    r1.text()
    conversation.responses.append(r1)

    r2 = conversation.prompt("world")
    r2.text()
    conversation.responses.append(r2)

    r1.log_to_db(db)
    r2.log_to_db(db)

    r1_id = r1.id
    r2_id = r2.id

    forked = conversation.fork(r1)
    r3 = forked.prompt("forked")
    r3.text()
    r3.log_to_db(db)

    r3_row = db["responses"].get(r3.id)
    assert r3_row["parent_response_id"] == r1_id

    r1_row = db["responses"].get(r1_id)
    assert r1_row["parent_response_id"] is None

    r2_row = db["responses"].get(r2_id)
    assert r2_row["parent_response_id"] is None


def test_logs_json_includes_parent_response_id(user_path):
    log_path = str(user_path / "logs_fork_json.db")
    db = sqlite_utils.Database(log_path)
    migrate(db)

    start = datetime.datetime.now(datetime.timezone.utc)

    conv1_id = "conv1"
    conv2_id = "conv2"

    db["conversations"].insert_all(
        [
            {"id": conv1_id, "name": "Original", "model": "davinci"},
            {"id": conv2_id, "name": "Forked", "model": "davinci"},
        ]
    )

    response1_id = str(monotonic_ulid()).lower()
    response2_id = str(monotonic_ulid()).lower()
    response3_id = str(monotonic_ulid()).lower()

    db["responses"].insert_all(
        [
            {
                "id": response1_id,
                "model": "davinci",
                "prompt": "hello",
                "response": "Hello!",
                "conversation_id": conv1_id,
                "datetime_utc": (start + datetime.timedelta(seconds=0)).isoformat(),
                "parent_response_id": None,
            },
            {
                "id": response2_id,
                "model": "davinci",
                "prompt": "how are you",
                "response": "Fine",
                "conversation_id": conv1_id,
                "datetime_utc": (start + datetime.timedelta(seconds=1)).isoformat(),
                "parent_response_id": None,
            },
            {
                "id": response3_id,
                "model": "davinci",
                "prompt": "different question",
                "response": "Forked answer",
                "conversation_id": conv2_id,
                "datetime_utc": (start + datetime.timedelta(seconds=2)).isoformat(),
                "parent_response_id": response1_id,
            },
        ]
    )

    runner = CliRunner()
    result = runner.invoke(
        cli, ["logs", "-p", log_path, "--json", "-n", "3"], catch_exceptions=False
    )
    assert result.exit_code == 0
    logs = json.loads(result.output)

    response3 = None
    for log in logs:
        if log["id"] == response3_id:
            response3 = log
            break

    assert response3 is not None
    assert response3["parent_response_id"] == response1_id


def test_logs_short_includes_parent_response(user_path):
    log_path = str(user_path / "logs_fork_short.db")
    db = sqlite_utils.Database(log_path)
    migrate(db)

    start = datetime.datetime.now(datetime.timezone.utc)

    conv1_id = "conv1"
    conv2_id = "conv2"

    db["conversations"].insert_all(
        [
            {"id": conv1_id, "name": "Original", "model": "davinci"},
            {"id": conv2_id, "name": "Forked", "model": "davinci"},
        ]
    )

    response1_id = str(monotonic_ulid()).lower()
    response3_id = str(monotonic_ulid()).lower()

    db["responses"].insert_all(
        [
            {
                "id": response1_id,
                "model": "davinci",
                "prompt": "hello",
                "response": "Hello!",
                "conversation_id": conv1_id,
                "datetime_utc": (start + datetime.timedelta(seconds=0)).isoformat(),
                "parent_response_id": None,
            },
            {
                "id": response3_id,
                "model": "davinci",
                "prompt": "different question",
                "response": "Forked answer",
                "conversation_id": conv2_id,
                "datetime_utc": (start + datetime.timedelta(seconds=2)).isoformat(),
                "parent_response_id": response1_id,
            },
        ]
    )

    runner = CliRunner()
    result = runner.invoke(
        cli, ["logs", "-p", log_path, "-s", "-n", "2"], catch_exceptions=False
    )
    assert result.exit_code == 0
    output = result.output

    # The forked response should show parent_response field
    assert f"parent_response: {response1_id}" in output


def test_logs_markdown_includes_fork_info(user_path):
    log_path = str(user_path / "logs_fork_md.db")
    db = sqlite_utils.Database(log_path)
    migrate(db)

    start = datetime.datetime.now(datetime.timezone.utc)

    conv1_id = "conv1"
    conv2_id = "conv2"

    db["conversations"].insert_all(
        [
            {"id": conv1_id, "name": "Original", "model": "davinci"},
            {"id": conv2_id, "name": "Forked", "model": "davinci"},
        ]
    )

    response1_id = str(monotonic_ulid()).lower()
    response3_id = str(monotonic_ulid()).lower()

    db["responses"].insert_all(
        [
            {
                "id": response1_id,
                "model": "davinci",
                "prompt": "hello",
                "response": "Hello!",
                "conversation_id": conv1_id,
                "datetime_utc": (start + datetime.timedelta(seconds=0)).isoformat(),
                "parent_response_id": None,
            },
            {
                "id": response3_id,
                "model": "davinci",
                "prompt": "different question",
                "response": "Forked answer",
                "conversation_id": conv2_id,
                "datetime_utc": (start + datetime.timedelta(seconds=2)).isoformat(),
                "parent_response_id": response1_id,
            },
        ]
    )

    runner = CliRunner()
    result = runner.invoke(
        cli, ["logs", "-p", log_path, "-n", "2"], catch_exceptions=False
    )
    assert result.exit_code == 0
    output = result.output

    assert "forked from response:" in output
    assert response1_id in output
