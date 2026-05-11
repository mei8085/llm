"""测试 Provider 适配器接口和错误归一化

本测试文件验证：
1. BaseProviderAdapter 基类的基本功能
2. AnthropicProviderAdapter 的消息处理和错误归一化
3. GeminiProviderAdapter 的消息处理和错误归一化
4. 统一接口的一致性
"""

import pytest
import json
from unittest.mock import MagicMock, patch

import llm
from llm import (
    AnthropicProviderAdapter,
    BaseProviderAdapter,
    GeminiProviderAdapter,
    ProviderAPIError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderErrorNormalizer,
    ProviderRateLimitError,
    ProviderTimeoutError,
    Response,
    Prompt,
)
from llm.parts import StreamEvent


class SimpleModel:
    """简单的模型类用于测试"""

    def __init__(self):
        self.calls = []

    def build_messages(self, prompt, conversation=None, image_detail=None):
        self.calls.append(("build_messages", prompt, conversation, image_detail))
        return [{"role": "user", "content": prompt.prompt}]

    def build_kwargs(self, prompt, stream):
        self.calls.append(("build_kwargs", prompt, stream))
        return {"temperature": 0.7}

    def get_client(self, key, async_=False):
        self.calls.append(("get_client", key, async_))
        return "mock_client"


class ConcreteTestAdapter(BaseProviderAdapter):
    """用于测试的具体适配器实现"""

    provider_name = "test"

    def process_streaming_response(
        self, response_object, response: Response
    ):
        for item in response_object:
            yield item

    def process_non_streaming_response(
        self, response_object, response: Response
    ):
        for item in response_object:
            yield item

    def normalize_error(self, exception: Exception) -> Exception:
        return exception


class TestBaseProviderAdapter:
    """测试 BaseProviderAdapter 基类"""

    def test_delegates_to_model_instance(self):
        """测试适配器委托调用到模型实例"""
        model = SimpleModel()
        adapter = ConcreteTestAdapter(model)

        mock_model = MagicMock()
        prompt = Prompt("test prompt", mock_model)
        messages = adapter.build_messages(prompt)
        assert messages == [{"role": "user", "content": "test prompt"}]

        kwargs = adapter.build_request_kwargs(prompt, stream=True)
        assert kwargs == {"temperature": 0.7}

        client = adapter.get_client("test_key", async_=False)
        assert client == "mock_client"

    def test_default_build_messages_without_model_method(self):
        """测试当模型没有 build_messages 方法时使用默认实现"""

        class ModelWithoutMethods:
            pass

        model = ModelWithoutMethods()
        adapter = ConcreteTestAdapter(model)

        mock_model = MagicMock()
        prompt = Prompt("test prompt", mock_model, system="system message")
        messages = adapter._default_build_messages(prompt)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "system message"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "test prompt"

    def test_default_build_messages_with_conversation(self):
        """测试默认消息构建器处理会话历史"""

        class ModelWithoutMethods:
            pass

        model = ModelWithoutMethods()
        adapter = ConcreteTestAdapter(model)

        mock_conversation = MagicMock()
        mock_prev_response = MagicMock()
        mock_prev_response.prompt.prompt = "previous question"
        mock_prev_response.text.return_value = "previous answer"
        mock_conversation.responses = [mock_prev_response]

        mock_model = MagicMock()
        prompt = Prompt("new question", mock_model)
        messages = adapter._default_build_messages(prompt, conversation=mock_conversation)

        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "previous question"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "previous answer"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "new question"


class TestAnthropicProviderAdapter:
    """测试 Anthropic 适配器"""

    def test_provider_name(self):
        adapter = AnthropicProviderAdapter(SimpleModel())
        assert adapter.provider_name == "anthropic"

    def test_process_non_streaming_text_response(self):
        """测试处理非流式文本响应"""
        adapter = AnthropicProviderAdapter(SimpleModel())

        mock_model = MagicMock()
        prompt = Prompt("test", mock_model)
        response = Response(prompt, MagicMock(), stream=False)

        mock_response_obj = MagicMock()
        mock_response_obj.content = [MagicMock(type="text", text="Hello, World!")]
        mock_response_obj.usage = MagicMock(input_tokens=10, output_tokens=5)

        events = list(adapter.process_non_streaming_response(mock_response_obj, response))

        assert len(events) == 1
        assert isinstance(events[0], StreamEvent)
        assert events[0].type == "text"
        assert events[0].chunk == "Hello, World!"
        assert response.input_tokens == 10
        assert response.output_tokens == 5

    def test_process_streaming_text_response(self):
        """测试处理流式文本响应"""
        adapter = AnthropicProviderAdapter(SimpleModel())

        mock_model = MagicMock()
        prompt = Prompt("test", mock_model)
        response = Response(prompt, MagicMock(), stream=True)

        def mock_stream():
            for i, text_chunk in enumerate(["Hello", ", ", "World", "!"]):
                event = MagicMock()
                event.type = "content_block_delta"
                event.delta = MagicMock(type="text_delta", text=text_chunk)
                yield event

            stop_event = MagicMock()
            stop_event.type = "message_stop"
            stop_event.message = MagicMock()
            stop_event.message.usage = MagicMock(input_tokens=15, output_tokens=8)
            yield stop_event

        events = list(adapter.process_streaming_response(mock_stream(), response))

        text_events = [e for e in events if e.type == "text"]
        assert len(text_events) == 4
        assert "".join(e.chunk for e in text_events) == "Hello, World!"
        assert response.input_tokens == 15
        assert response.output_tokens == 8

    def test_process_non_streaming_tool_call_with_empty_args(self):
        """测试处理无参工具调用（非流式）"""
        adapter = AnthropicProviderAdapter(SimpleModel())

        mock_model = MagicMock()
        prompt = Prompt("test", mock_model)
        response = Response(prompt, MagicMock(), stream=False)

        mock_response_obj = MagicMock()
        mock_tool_block = MagicMock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.id = "tool_123"
        mock_tool_block.name = "get_current_time"
        mock_tool_block.input = {}
        mock_response_obj.content = [mock_tool_block]
        mock_response_obj.usage = MagicMock(input_tokens=5, output_tokens=3)

        list(adapter.process_non_streaming_response(mock_response_obj, response))

        assert response._tool_calls
        tool_calls = response._tool_calls
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_current_time"
        assert tool_calls[0].arguments == {}
        assert tool_calls[0].tool_call_id == "tool_123"

    def test_process_streaming_tool_call_with_empty_args(self):
        """测试处理无参工具调用（流式）"""
        adapter = AnthropicProviderAdapter(SimpleModel())

        mock_model = MagicMock()
        prompt = Prompt("test", mock_model)
        response = Response(prompt, MagicMock(), stream=True)

        def mock_stream():
            start_event = MagicMock()
            start_event.type = "content_block_start"
            start_event.index = 0
            content_block = MagicMock()
            content_block.type = "tool_use"
            content_block.id = "tool_456"
            content_block.name = "get_weather"
            start_event.content_block = content_block
            yield start_event

            stop_event = MagicMock()
            stop_event.type = "message_stop"
            stop_event.message = MagicMock()
            stop_event.message.usage = MagicMock(input_tokens=10, output_tokens=5)
            yield stop_event

        list(adapter.process_streaming_response(mock_stream(), response))

        assert response._tool_calls
        tool_calls = response._tool_calls
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"
        assert tool_calls[0].arguments == {}
        assert tool_calls[0].tool_call_id == "tool_456"


class TestGeminiProviderAdapter:
    """测试 Gemini 适配器"""

    def test_provider_name(self):
        adapter = GeminiProviderAdapter(SimpleModel())
        assert adapter.provider_name == "gemini"

    def test_process_non_streaming_text_response(self):
        """测试处理非流式文本响应"""
        adapter = GeminiProviderAdapter(SimpleModel())

        mock_model = MagicMock()
        prompt = Prompt("test", mock_model)
        response = Response(prompt, MagicMock(), stream=False)

        mock_candidate = MagicMock()
        mock_content = MagicMock()
        mock_part = MagicMock(text="Hello from Gemini!", function_call=None)
        mock_content.parts = [mock_part]
        mock_candidate.content = mock_content
        mock_candidate.usage_metadata = MagicMock(total_token_count=20)

        mock_response_obj = MagicMock()
        mock_response_obj.candidates = [mock_candidate]

        events = list(adapter.process_non_streaming_response(mock_response_obj, response))

        assert len(events) == 1
        assert isinstance(events[0], StreamEvent)
        assert events[0].type == "text"
        assert events[0].chunk == "Hello from Gemini!"
        assert response.input_tokens == 20

    def test_process_streaming_text_response(self):
        """测试处理流式文本响应"""
        adapter = GeminiProviderAdapter(SimpleModel())

        mock_model = MagicMock()
        prompt = Prompt("test", mock_model)
        response = Response(prompt, MagicMock(), stream=True)

        def mock_stream():
            for text_chunk in ["Hello", " from ", "Gemini", "!"]:
                chunk = MagicMock()
                mock_candidate = MagicMock()
                mock_content = MagicMock()
                mock_part = MagicMock(text=text_chunk, function_call=None)
                mock_content.parts = [mock_part]
                mock_candidate.content = mock_content
                mock_candidate.usage_metadata = None
                chunk.candidates = [mock_candidate]
                yield chunk

        events = list(adapter.process_streaming_response(mock_stream(), response))

        text_events = [e for e in events if e.type == "text"]
        assert len(text_events) == 4
        assert "".join(e.chunk for e in text_events) == "Hello from Gemini!"

    def test_process_non_streaming_tool_call_with_empty_args(self):
        """测试处理无参工具调用（非流式）"""
        adapter = GeminiProviderAdapter(SimpleModel())

        mock_model = MagicMock()
        prompt = Prompt("test", mock_model)
        response = Response(prompt, MagicMock(), stream=False)

        mock_candidate = MagicMock()
        mock_content = MagicMock()
        mock_function_call = MagicMock()
        mock_function_call.name = "get_current_time"
        mock_function_call.args = {}
        mock_part = MagicMock(text=None, function_call=mock_function_call)
        mock_content.parts = [mock_part]
        mock_candidate.content = mock_content
        mock_candidate.usage_metadata = MagicMock(total_token_count=10)

        mock_response_obj = MagicMock()
        mock_response_obj.candidates = [mock_candidate]

        list(adapter.process_non_streaming_response(mock_response_obj, response))

        assert response._tool_calls
        tool_calls = response._tool_calls
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_current_time"
        assert tool_calls[0].arguments == {}

    def test_process_streaming_tool_call_with_empty_args(self):
        """测试处理无参工具调用（流式）"""
        adapter = GeminiProviderAdapter(SimpleModel())

        mock_model = MagicMock()
        prompt = Prompt("test", mock_model)
        response = Response(prompt, MagicMock(), stream=True)

        def mock_stream():
            chunk = MagicMock()
            mock_candidate = MagicMock()
            mock_content = MagicMock()
            mock_function_call = MagicMock()
            mock_function_call.name = "get_weather"
            mock_function_call.args = {}
            mock_part = MagicMock(text=None, function_call=mock_function_call)
            mock_content.parts = [mock_part]
            mock_candidate.content = mock_content
            mock_candidate.usage_metadata = None
            chunk.candidates = [mock_candidate]
            yield chunk

        list(adapter.process_streaming_response(mock_stream(), response))

        assert response._tool_calls
        tool_calls = response._tool_calls
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"
        assert tool_calls[0].arguments == {}


class TestErrorNormalization:
    """测试错误归一化"""

    def test_openai_error_normalization(self):
        """测试 OpenAI 错误归一化"""
        import openai

        auth_error = openai.AuthenticationError(
            message="Invalid API key",
            response=MagicMock(),
            body=None,
        )
        normalized = ProviderErrorNormalizer.normalize_openai_error(auth_error)
        assert isinstance(normalized, ProviderAuthenticationError)

        rate_error = openai.RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(),
            body=None,
        )
        normalized = ProviderErrorNormalizer.normalize_openai_error(rate_error)
        assert isinstance(normalized, ProviderRateLimitError)

        api_error = openai.APIError(
            message="API error",
            request=MagicMock(),
            body=None,
        )
        normalized = ProviderErrorNormalizer.normalize_openai_error(api_error)
        assert isinstance(normalized, ProviderAPIError)

        timeout_error = openai.APITimeoutError(request=MagicMock())
        normalized = ProviderErrorNormalizer.normalize_openai_error(timeout_error)
        assert isinstance(normalized, ProviderTimeoutError)

        conn_error = openai.APIConnectionError(request=MagicMock())
        normalized = ProviderErrorNormalizer.normalize_openai_error(conn_error)
        assert isinstance(normalized, ProviderConnectionError)

    def test_anthropic_error_normalization(self):
        """测试 Anthropic 错误归一化（当 SDK 可用时）"""
        try:
            import anthropic

            auth_error = anthropic.AuthenticationError(
                message="Invalid API key",
                body={"error": "auth_failed"},
            )
            normalized = ProviderErrorNormalizer.normalize_anthropic_error(auth_error)
            assert isinstance(normalized, ProviderAuthenticationError)

            rate_error = anthropic.RateLimitError(
                message="Rate limit exceeded",
                body={"error": "rate_limit"},
            )
            normalized = ProviderErrorNormalizer.normalize_anthropic_error(rate_error)
            assert isinstance(normalized, ProviderRateLimitError)

            api_error = anthropic.APIError(
                message="API error",
                body={"error": "api_error"},
            )
            normalized = ProviderErrorNormalizer.normalize_anthropic_error(api_error)
            assert isinstance(normalized, ProviderAPIError)

            timeout_error = anthropic.APITimeoutError(request=MagicMock())
            normalized = ProviderErrorNormalizer.normalize_anthropic_error(timeout_error)
            assert isinstance(normalized, ProviderTimeoutError)

            conn_error = anthropic.APIConnectionError(request=MagicMock())
            normalized = ProviderErrorNormalizer.normalize_anthropic_error(conn_error)
            assert isinstance(normalized, ProviderConnectionError)

        except ImportError:
            pytest.skip("Anthropic SDK not installed")

    def test_gemini_error_normalization(self):
        """测试 Gemini 错误归一化（当 SDK 可用时）"""
        try:
            import google.api_core.exceptions as google_exceptions

            auth_error = google_exceptions.Unauthenticated("Invalid credentials")
            normalized = ProviderErrorNormalizer.normalize_gemini_error(auth_error)
            assert isinstance(normalized, ProviderAuthenticationError)

            rate_error = google_exceptions.ResourceExhausted("Quota exceeded")
            normalized = ProviderErrorNormalizer.normalize_gemini_error(rate_error)
            assert isinstance(normalized, ProviderRateLimitError)

            timeout_error = google_exceptions.DeadlineExceeded("Request timed out")
            normalized = ProviderErrorNormalizer.normalize_gemini_error(timeout_error)
            assert isinstance(normalized, ProviderTimeoutError)

            api_error = google_exceptions.ServiceUnavailable("Service unavailable")
            normalized = ProviderErrorNormalizer.normalize_gemini_error(api_error)
            assert isinstance(normalized, ProviderAPIError)

        except ImportError:
            pytest.skip("Google API core SDK not installed")

    def test_unknown_error_passthrough(self):
        """测试未知错误直接透传"""
        original_error = ValueError("Some random error")
        normalized = ProviderErrorNormalizer.normalize_openai_error(original_error)
        assert normalized is original_error


class TestAdapterIntegration:
    """测试适配器集成到模型中"""

    def test_anthropic_model_with_adapter(self):
        """测试模型使用 Anthropic 适配器"""

        class MockAnthropicModel:
            def __init__(self):
                self.model_id = "claude-sonnet"
                self.needs_key = "anthropic"
                self.key_env_var = "ANTHROPIC_API_KEY"

            def _get_provider_adapter(self):
                return AnthropicProviderAdapter(self)

            def build_messages(self, prompt, conversation=None, image_detail=None):
                return [{"role": "user", "content": prompt.prompt}]

            def build_kwargs(self, prompt, stream):
                return {"max_tokens": 1000}

            def get_client(self, key, async_=False):
                return MagicMock()

        model = MockAnthropicModel()
        adapter = model._get_provider_adapter()

        assert adapter.provider_name == "anthropic"

        mock_model = MagicMock()
        prompt = Prompt("test", mock_model)
        messages = adapter.build_messages(prompt)
        assert messages == [{"role": "user", "content": "test"}]

    def test_gemini_model_with_adapter(self):
        """测试模型使用 Gemini 适配器"""

        class MockGeminiModel:
            def __init__(self):
                self.model_id = "gemini-pro"
                self.needs_key = "gemini"
                self.key_env_var = "GOOGLE_API_KEY"

            def _get_provider_adapter(self):
                return GeminiProviderAdapter(self)

            def build_messages(self, prompt, conversation=None, image_detail=None):
                return [{"role": "user", "parts": [{"text": prompt.prompt}]}]

            def build_kwargs(self, prompt, stream):
                return {"temperature": 0.8}

            def get_client(self, key, async_=False):
                return MagicMock()

        model = MockGeminiModel()
        adapter = model._get_provider_adapter()

        assert adapter.provider_name == "gemini"

        mock_model = MagicMock()
        prompt = Prompt("test", mock_model)
        messages = adapter.build_messages(prompt)
        assert messages == [{"role": "user", "parts": [{"text": "test"}]}]


class TestUnifiedInterface:
    """测试统一接口的一致性"""

    def test_all_adapters_have_same_methods(self):
        """测试所有适配器具有相同的接口方法"""
        from llm import ProviderAdapter

        required_methods = [
            "build_messages",
            "build_request_kwargs",
            "get_client",
            "process_streaming_response",
            "process_non_streaming_response",
            "normalize_error",
            "set_usage",
        ]

        for adapter_class in [
            AnthropicProviderAdapter,
            GeminiProviderAdapter,
        ]:
            for method in required_methods:
                assert hasattr(adapter_class, method), (
                    f"{adapter_class.__name__} missing method: {method}"
                )

    def test_all_adapters_have_provider_name(self):
        """测试所有适配器都定义了 provider_name"""
        for adapter_class in [
            AnthropicProviderAdapter,
            GeminiProviderAdapter,
        ]:
            assert hasattr(adapter_class, "provider_name"), (
                f"{adapter_class.__name__} missing provider_name"
            )
            assert isinstance(adapter_class.provider_name, str)
            assert adapter_class.provider_name != "unknown"

    def test_error_hierarchy(self):
        """测试错误类型的层级关系"""
        from llm import (
            ProviderAPIError,
            ProviderAuthenticationError,
            ProviderRateLimitError,
            ModelError,
        )

        assert issubclass(ProviderAPIError, ModelError)
        assert issubclass(ProviderAuthenticationError, ProviderAPIError)
        assert issubclass(ProviderRateLimitError, ProviderAPIError)
