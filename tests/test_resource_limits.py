import asyncio
import json
import llm
from llm import ResourceLimits, ResourceLimitExceeded
import pytest
import sqlite_utils
import time


class TestResourceLimits:
    def test_resource_limits_defaults(self):
        limits = ResourceLimits()
        assert limits.http_requests is None
        assert limits.tokens is None
        assert limits.tool_calls is None
        assert limits.execution_time_ms is None
        assert not limits.has_any_limit()

    def test_resource_limits_with_values(self):
        limits = ResourceLimits(
            http_requests=10,
            tokens=1000,
            tool_calls=5,
            execution_time_ms=5000,
        )
        assert limits.http_requests == 10
        assert limits.tokens == 1000
        assert limits.tool_calls == 5
        assert limits.execution_time_ms == 5000
        assert limits.has_any_limit()

    def test_resource_limits_partial(self):
        limits = ResourceLimits(http_requests=10)
        assert limits.http_requests == 10
        assert limits.has_any_limit()


class TestResourceTracker:
    def test_tracker_no_limits(self):
        tracker = llm.ResourceTracker()
        usage = tracker.usage
        assert usage.http_requests == 0
        assert usage.tokens == 0
        assert usage.tool_calls == 0
        assert usage.execution_time_ms == 0
        assert tracker.check_pre_call() is None

    def test_can_consume_http_requests(self):
        tracker = llm.ResourceTracker(ResourceLimits(http_requests=2))
        assert tracker.can_consume_http_requests(1)
        assert tracker.can_consume_http_requests(2)
        assert not tracker.can_consume_http_requests(3)

    def test_can_consume_tokens(self):
        tracker = llm.ResourceTracker(ResourceLimits(tokens=100))
        assert tracker.can_consume_tokens(50)
        assert tracker.can_consume_tokens(100)
        assert not tracker.can_consume_tokens(101)

    def test_can_consume_tool_calls(self):
        tracker = llm.ResourceTracker(ResourceLimits(tool_calls=2))
        assert tracker.can_consume_tool_calls(1)
        assert tracker.can_consume_tool_calls(2)
        assert not tracker.can_consume_tool_calls(3)

    def test_tracker_increment_tool_calls(self):
        tracker = llm.ResourceTracker(ResourceLimits(tool_calls=2))
        exceeded, alert = tracker.increment_tool_calls()
        assert not exceeded
        assert alert is None

        exceeded, alert = tracker.increment_tool_calls()
        assert not exceeded
        assert alert is None

        exceeded, alert = tracker.increment_tool_calls()
        assert exceeded
        assert "Tool call limit exceeded: 3 > 2" in alert

    def test_tracker_increment_tokens(self):
        tracker = llm.ResourceTracker(ResourceLimits(tokens=100))
        exceeded, alert = tracker.increment_tokens(50)
        assert not exceeded
        assert alert is None

        exceeded, alert = tracker.increment_tokens(50)
        assert not exceeded
        assert alert is None

        exceeded, alert = tracker.increment_tokens(1)
        assert exceeded
        assert "Token limit exceeded: 101 > 100" in alert

    def test_tracker_increment_http_requests(self):
        tracker = llm.ResourceTracker(ResourceLimits(http_requests=2))
        exceeded, alert = tracker.increment_http_requests()
        assert not exceeded
        assert alert is None

        exceeded, alert = tracker.increment_http_requests()
        assert not exceeded
        assert alert is None

        exceeded, alert = tracker.increment_http_requests()
        assert exceeded
        assert "HTTP request limit exceeded: 3 > 2" in alert

    def test_tracker_add_execution_time(self):
        tracker = llm.ResourceTracker(ResourceLimits(execution_time_ms=100))
        exceeded, alert = tracker.add_execution_time(50)
        assert not exceeded
        assert alert is None

        exceeded, alert = tracker.add_execution_time(50)
        assert not exceeded
        assert alert is None

        exceeded, alert = tracker.add_execution_time(1)
        assert exceeded
        assert "Execution time limit exceeded: 101ms > 100ms" in alert

    def test_tracker_check_pre_call(self):
        tracker = llm.ResourceTracker(ResourceLimits(tool_calls=1))
        tracker.increment_tool_calls()
        pre_check = tracker.check_pre_call()
        assert pre_check is not None
        assert "tool_calls already at limit" in pre_check

    def test_tracker_check_pre_call_includes_http_and_tokens(self):
        tracker = llm.ResourceTracker(ResourceLimits(http_requests=1, tokens=100))
        tracker.increment_http_requests()
        pre_check = tracker.check_pre_call()
        assert pre_check is not None
        assert "http_requests already at limit" in pre_check

        tracker2 = llm.ResourceTracker(ResourceLimits(tokens=100))
        tracker2.increment_tokens(100)
        pre_check2 = tracker2.check_pre_call()
        assert pre_check2 is not None
        assert "tokens already at limit" in pre_check2

    def test_tracker_alert_only_once(self):
        tracker = llm.ResourceTracker(ResourceLimits(tool_calls=1))
        tracker.increment_tool_calls()
        exceeded1, alert1 = tracker.increment_tool_calls()
        exceeded2, alert2 = tracker.increment_tool_calls()

        assert exceeded1
        assert exceeded2
        assert alert1 is not None
        assert alert2 is None

    def test_tracker_clear_alerts(self):
        tracker = llm.ResourceTracker(ResourceLimits(tool_calls=1))
        tracker.increment_tool_calls()
        tracker.increment_tool_calls()
        tracker.clear_alerts()

        exceeded, alert = tracker.increment_tool_calls()
        assert exceeded
        assert alert is not None

    def test_tracker_thread_safe(self):
        tracker = llm.ResourceTracker(ResourceLimits(tokens=1000))

        def increment_many():
            for _ in range(100):
                tracker.increment_tokens(1)

        import threading

        threads = [threading.Thread(target=increment_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert tracker.usage.tokens == 1000


class TestToolResourceLimits:
    def test_tool_with_resource_limits(self):
        def my_tool():
            return "result"

        limits = ResourceLimits(tool_calls=3)
        tool = llm.Tool.function(my_tool, resource_limits=limits)

        assert tool.resource_limits is not None
        assert tool.resource_tracker is not None
        assert tool.resource_limits.tool_calls == 3

    def test_tool_without_resource_limits(self):
        def my_tool():
            return "result"

        tool = llm.Tool.function(my_tool)

        assert tool.resource_limits is None
        assert tool.resource_tracker is None

    def test_tool_with_empty_limits(self):
        def my_tool():
            return "result"

        limits = ResourceLimits()
        tool = llm.Tool.function(my_tool, resource_limits=limits)

        assert tool.resource_limits is not None
        assert tool.resource_tracker is None
        assert not tool.resource_limits.has_any_limit()

    def test_tool_report_http_requests(self):
        limits = ResourceLimits(http_requests=2)
        tool = llm.Tool("test_tool", resource_limits=limits)

        exceeded, alert = tool.report_http_requests(1)
        assert not exceeded
        assert alert is None
        assert tool.resource_tracker.usage.http_requests == 1

        exceeded, alert = tool.report_http_requests(1)
        assert not exceeded
        assert alert is None

        exceeded, alert = tool.report_http_requests(1)
        assert exceeded
        assert "HTTP request limit exceeded" in alert

    def test_tool_report_tokens(self):
        limits = ResourceLimits(tokens=100)
        tool = llm.Tool("test_tool", resource_limits=limits)

        exceeded, alert = tool.report_tokens(50)
        assert not exceeded
        assert alert is None
        assert tool.resource_tracker.usage.tokens == 50

        exceeded, alert = tool.report_tokens(60)
        assert exceeded
        assert "Token limit exceeded" in alert

    def test_tool_can_make_http_request(self):
        limits = ResourceLimits(http_requests=2)
        tool = llm.Tool("test_tool", resource_limits=limits)

        assert tool.can_make_http_request(1)
        assert tool.can_make_http_request(2)
        assert not tool.can_make_http_request(3)

    def test_tool_can_use_tokens(self):
        limits = ResourceLimits(tokens=100)
        tool = llm.Tool("test_tool", resource_limits=limits)

        assert tool.can_use_tokens(50)
        assert tool.can_use_tokens(100)
        assert not tool.can_use_tokens(101)

    def test_tool_without_limits_allows_all(self):
        tool = llm.Tool("test_tool")

        assert tool.can_make_http_request(1000)
        assert tool.can_use_tokens(1000000)
        exceeded, alert = tool.report_http_requests(100)
        assert not exceeded
        exceeded, alert = tool.report_tokens(100000)
        assert not exceeded


class TestHttpAndTokenTracking:
    def test_plugin_reports_http_requests(self):
        alerts = []

        def alert_logger(message, tool_obj):
            alerts.append((message, tool_obj.name))

        http_calls_made = []

        def plugin_with_http(tool_instance):
            for i in range(3):
                if not tool_instance.can_make_http_request(1):
                    return f"Cannot make more HTTP calls, already at limit"
                http_calls_made.append(i)
                exceeded, alert = tool_instance.report_http_requests(1)
                if exceeded and alert:
                    alert_logger(alert, tool_instance)
            return f"Made {len(http_calls_made)} HTTP calls"

        limits = ResourceLimits(http_requests=3)
        tool = llm.Tool.function(plugin_with_http, resource_limits=limits)

        result = tool.implementation(tool)

        assert len(http_calls_made) == 3
        assert result == "Made 3 HTTP calls"

        exceeded, alert = tool.report_http_requests(1)
        if exceeded and alert:
            alert_logger(alert, tool)

        assert len(alerts) == 1
        assert "HTTP request limit exceeded" in alerts[0][0]

    def test_plugin_reports_tokens(self):
        alerts = []

        def alert_logger(message, tool_obj):
            alerts.append((message, tool_obj.name))

        def plugin_with_tokens(tool_instance):
            total_tokens = 0
            batch_sizes = [30, 40, 50]
            for batch in batch_sizes:
                if not tool_instance.can_use_tokens(batch):
                    return f"Cannot consume {batch} tokens, would exceed limit. Used: {total_tokens}"
                exceeded, alert = tool_instance.report_tokens(batch)
                total_tokens += batch
                if exceeded and alert:
                    alert_logger(alert, tool_instance)
            return f"Used {total_tokens} tokens"

        limits = ResourceLimits(tokens=100)
        tool = llm.Tool.function(plugin_with_tokens, resource_limits=limits)

        result = tool.implementation(tool)

        assert "Cannot consume 50 tokens" in result
        assert len(alerts) == 0
        assert tool.resource_tracker.usage.tokens == 70

    def test_plugin_reports_http_and_tokens(self):
        alerts = []

        def alert_logger(message, tool_obj):
            alerts.append((message, tool_obj.name))

        def hybrid_plugin(tool_instance):
            output = []
            for i in range(3):
                if not tool_instance.can_make_http_request(1):
                    output.append(f"HTTP limit reached at call {i+1}")
                    break

                if not tool_instance.can_use_tokens(50):
                    output.append(f"Token limit reached at call {i+1}")
                    break

                exceeded_http, alert_http = tool_instance.report_http_requests(1)
                if exceeded_http and alert_http:
                    alert_logger(alert_http, tool_instance)

                exceeded_tokens, alert_tokens = tool_instance.report_tokens(50)
                if exceeded_tokens and alert_tokens:
                    alert_logger(alert_tokens, tool_instance)

                output.append(f"Call {i+1} complete")

            return ", ".join(output)

        limits = ResourceLimits(http_requests=3, tokens=120)
        tool = llm.Tool.function(hybrid_plugin, resource_limits=limits)

        result = tool.implementation(tool)

        assert "Token limit reached at call 3" in result
        assert tool.resource_tracker.usage.http_requests == 2
        assert tool.resource_tracker.usage.tokens == 100


class TestExecuteToolCallsResourceLimits:
    def test_tool_call_within_limits(self):
        model = llm.get_model("echo")
        alerts = []

        def limited_tool():
            return "success"

        limits = ResourceLimits(tool_calls=2)
        tool = llm.Tool.function(limited_tool, resource_limits=limits)

        chain_response = model.chain(
            json.dumps({"tool_calls": [{"name": "limited_tool"}]}),
            tools=[tool],
        )
        output = chain_response.text()

        assert '"output": "success"' in output
        assert tool.resource_tracker.usage.tool_calls == 1

    def test_tool_call_limit_exceeded_downgrade(self):
        model = llm.get_model("echo")
        alerts_logged = []

        def alert_logger(message, tool_obj):
            alerts_logged.append((message, tool_obj.name))

        def limited_tool():
            return "success"

        limits = ResourceLimits(tool_calls=1)
        tool = llm.Tool.function(limited_tool, resource_limits=limits)

        chain1 = model.chain(
            json.dumps({"tool_calls": [{"name": "limited_tool"}]}),
            tools=[tool],
        )
        output1 = chain1.text()
        assert '"output": "success"' in output1

        chain2 = model.chain(
            json.dumps({"tool_calls": [{"name": "limited_tool"}]}),
            tools=[tool],
        )
        output2 = chain2.text()

        assert "Resource limit reached" in output2
        assert "tool_calls already at limit" in output2
        assert tool.resource_tracker.usage.tool_calls == 1

    def test_http_limit_exceeded_pre_call_check(self):
        model = llm.get_model("echo")

        def my_tool():
            return "success"

        limits = ResourceLimits(http_requests=1)
        tool = llm.Tool.function(my_tool, resource_limits=limits)

        tool.resource_tracker.increment_http_requests()

        chain_response = model.chain(
            json.dumps({"tool_calls": [{"name": "my_tool"}]}),
            tools=[tool],
        )
        output = chain_response.text()

        assert "Resource limit reached" in output
        assert "http_requests already at limit" in output

    def test_token_limit_exceeded_pre_call_check(self):
        model = llm.get_model("echo")

        def my_tool():
            return "success"

        limits = ResourceLimits(tokens=100)
        tool = llm.Tool.function(my_tool, resource_limits=limits)

        tool.resource_tracker.increment_tokens(100)

        chain_response = model.chain(
            json.dumps({"tool_calls": [{"name": "my_tool"}]}),
            tools=[tool],
        )
        output = chain_response.text()

        assert "Resource limit reached" in output
        assert "tokens already at limit" in output

    def test_alert_logger_called_on_exceed(self):
        model = llm.get_model("echo")
        alerts = []

        def alert_logger(message, tool_obj):
            alerts.append((message, tool_obj.name))

        def limited_tool():
            return "success"

        limits = ResourceLimits(tool_calls=1)
        tool = llm.Tool.function(limited_tool, resource_limits=limits)

        chain1 = model.chain(
            json.dumps({"tool_calls": [{"name": "limited_tool"}]}),
            tools=[tool],
        )
        output1 = chain1.text()

        chain2 = model.chain(
            json.dumps({"tool_calls": [{"name": "limited_tool"}]}),
            tools=[tool],
        )
        output2 = chain2.text()

        assert tool.resource_tracker.usage.tool_calls == 1
        assert "Resource limit reached" in output2

    def test_execution_time_limit(self):
        model = llm.get_model("echo")

        def slow_tool():
            time.sleep(0.1)
            return "slow result"

        limits = ResourceLimits(execution_time_ms=50)
        tool = llm.Tool.function(slow_tool, resource_limits=limits)

        chain_response = model.chain(
            json.dumps({"tool_calls": [{"name": "slow_tool"}]}),
            tools=[tool],
        )
        output = chain_response.text()

        assert tool.resource_tracker.usage.execution_time_ms >= 100


class TestAsyncResourceLimits:
    @pytest.mark.asyncio
    async def test_async_tool_call_within_limits(self):
        model = llm.get_async_model("echo")

        async def async_limited_tool():
            await asyncio.sleep(0)
            return "async success"

        limits = ResourceLimits(tool_calls=2)
        tool = llm.Tool.function(async_limited_tool, resource_limits=limits)

        chain_response = model.chain(
            json.dumps({"tool_calls": [{"name": "async_limited_tool"}]}),
            tools=[tool],
        )
        output = await chain_response.text()

        assert '"output": "async success"' in output
        assert tool.resource_tracker.usage.tool_calls == 1

    @pytest.mark.asyncio
    async def test_async_tool_call_limit_exceeded(self):
        model = llm.get_async_model("echo")

        async def async_limited_tool():
            await asyncio.sleep(0)
            return "async success"

        limits = ResourceLimits(tool_calls=1)
        tool = llm.Tool.function(async_limited_tool, resource_limits=limits)

        chain1 = model.chain(
            json.dumps({"tool_calls": [{"name": "async_limited_tool"}]}),
            tools=[tool],
        )
        output1 = await chain1.text()
        assert '"output": "async success"' in output1

        chain2 = model.chain(
            json.dumps({"tool_calls": [{"name": "async_limited_tool"}]}),
            tools=[tool],
        )
        output2 = await chain2.text()

        assert "Resource limit reached" in output2
        assert tool.resource_tracker.usage.tool_calls == 1

    @pytest.mark.asyncio
    async def test_async_http_limit_pre_call_check(self):
        model = llm.get_async_model("echo")

        async def my_async_tool():
            await asyncio.sleep(0)
            return "success"

        limits = ResourceLimits(http_requests=1)
        tool = llm.Tool.function(my_async_tool, resource_limits=limits)

        tool.resource_tracker.increment_http_requests()

        chain_response = model.chain(
            json.dumps({"tool_calls": [{"name": "my_async_tool"}]}),
            tools=[tool],
        )
        output = await chain_response.text()

        assert "Resource limit reached" in output


class TestResourceLimitExceededException:
    def test_exception_with_details(self):
        exc = ResourceLimitExceeded(
            "Tool calls limit exceeded",
            resource_type="tool_calls",
            limit=5,
            usage=10,
        )
        assert exc.resource_type == "tool_calls"
        assert exc.limit == 5
        assert exc.usage == 10
        assert "Tool calls limit exceeded" in str(exc)

    def test_exception_basic(self):
        exc = ResourceLimitExceeded("Limit exceeded")
        assert exc.resource_type is None
        assert exc.limit is None
        assert exc.usage is None


class TestAlertLogging:
    def test_log_resource_alert(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))

        llm.log_resource_alert(
            tool_name="test_tool",
            tool_plugin="test_plugin",
            message="Test alert message",
            resource_type="http_requests",
            limit_value=10,
            usage_value=15,
        )

        db_path = tmp_path / "logs.db"
        assert db_path.exists()

        db = sqlite_utils.Database(db_path)
        alerts = list(db["resource_alerts"].rows)
        assert len(alerts) == 1
        alert = alerts[0]
        assert alert["tool_name"] == "test_tool"
        assert alert["tool_plugin"] == "test_plugin"
        assert alert["message"] == "Test alert message"
        assert alert["resource_type"] == "http_requests"
        assert alert["limit_value"] == 10
        assert alert["usage_value"] == 15

    def test_default_alert_logger_http(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))

        tool = llm.Tool(
            name="test_tool",
            plugin="test_plugin",
            resource_limits=ResourceLimits(http_requests=2),
        )

        llm.default_alert_logger(
            "HTTP request limit exceeded: 3 > 2", tool
        )

        db_path = tmp_path / "logs.db"
        db = sqlite_utils.Database(db_path)
        alerts = list(db["resource_alerts"].rows)
        assert len(alerts) == 1
        alert = alerts[0]
        assert alert["resource_type"] == "http_requests"
        assert alert["limit_value"] == 2
        assert alert["usage_value"] == 3

    def test_default_alert_logger_tokens(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))

        tool = llm.Tool(
            name="token_tool",
            plugin=None,
            resource_limits=ResourceLimits(tokens=100),
        )

        llm.default_alert_logger(
            "Token limit exceeded: 150 > 100", tool
        )

        db_path = tmp_path / "logs.db"
        db = sqlite_utils.Database(db_path)
        alerts = list(db["resource_alerts"].rows)
        assert len(alerts) == 1
        alert = alerts[0]
        assert alert["resource_type"] == "tokens"
        assert alert["limit_value"] == 100
        assert alert["usage_value"] == 150

    def test_default_alert_logger_tool_calls(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))

        tool = llm.Tool(
            name="limited_tool",
            plugin="my_plugin",
            resource_limits=ResourceLimits(tool_calls=5),
        )

        llm.default_alert_logger(
            "Tool call limit exceeded: 6 > 5", tool
        )

        db_path = tmp_path / "logs.db"
        db = sqlite_utils.Database(db_path)
        alerts = list(db["resource_alerts"].rows)
        assert len(alerts) == 1
        assert alerts[0]["resource_type"] == "tool_calls"

    def test_default_alert_logger_execution_time(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))

        tool = llm.Tool(
            name="slow_tool",
            resource_limits=ResourceLimits(execution_time_ms=1000),
        )

        llm.default_alert_logger(
            "Execution time limit exceeded: 1500ms > 1000ms", tool
        )

        db_path = tmp_path / "logs.db"
        db = sqlite_utils.Database(db_path)
        alerts = list(db["resource_alerts"].rows)
        assert len(alerts) == 1
        assert alerts[0]["resource_type"] == "execution_time_ms"
        assert alerts[0]["usage_value"] == 1500
        assert alerts[0]["limit_value"] == 1000


class TestFullWorkflow:
    def test_plugin_reports_http_and_triggers_alert_in_db(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))

        def plugin_with_http(tool_instance):
            result = []
            for i in range(5):
                exceeded, alert = tool_instance.report_http_requests(1)
                if exceeded and alert:
                    llm.default_alert_logger(alert, tool_instance)
                    result.append(f"Stopped at {i+1} due to HTTP limit")
                    break
                result.append(f"HTTP call {i+1}")
            return "; ".join(result)

        limits = ResourceLimits(http_requests=2)
        tool = llm.Tool.function(plugin_with_http, resource_limits=limits)

        output = tool.implementation(tool)

        assert "Stopped at 3 due to HTTP limit" in output
        assert tool.resource_tracker.usage.http_requests == 3

        db_path = tmp_path / "logs.db"
        db = sqlite_utils.Database(db_path)
        alerts = list(db["resource_alerts"].rows)
        assert len(alerts) == 1
        assert alerts[0]["resource_type"] == "http_requests"
        assert alerts[0]["usage_value"] == 3
        assert alerts[0]["limit_value"] == 2

    def test_plugin_reports_tokens_and_triggers_alert(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LLM_USER_PATH", str(tmp_path))

        def plugin_with_tokens(tool_instance):
            total = 0
            batches = [30, 40, 50, 60]
            for batch in batches:
                if not tool_instance.can_use_tokens(batch):
                    return f"Cannot use {batch} more tokens, total used: {total}"
                exceeded, alert = tool_instance.report_tokens(batch)
                total += batch
                if exceeded and alert:
                    llm.default_alert_logger(alert, tool_instance)
            return f"Used {total} tokens"

        limits = ResourceLimits(tokens=100)
        tool = llm.Tool.function(plugin_with_tokens, resource_limits=limits)

        output = tool.implementation(tool)

        assert "Cannot use 50 more tokens" in output
        assert tool.resource_tracker.usage.tokens == 70
