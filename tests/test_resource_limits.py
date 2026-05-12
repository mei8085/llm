import asyncio
import json
import llm
from llm import ResourceLimits, ResourceLimitExceeded
import pytest
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
