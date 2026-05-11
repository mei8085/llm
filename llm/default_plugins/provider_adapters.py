"""
统一的 Provider 适配器基类和示例实现

本模块提供了：
1. BaseProviderAdapter - 简化适配器实现的基类
2. AnthropicProviderAdapter - Anthropic 示例适配器
3. GeminiProviderAdapter - Gemini 示例适配器

外部插件可以直接使用这些基类或示例实现来快速实现统一的 provider 接口。
"""

from llm import (
    AsyncConversation,
    AsyncKeyModel,
    AsyncProviderAdapter,
    AsyncResponse,
    Conversation,
    KeyModel,
    ProviderAdapter,
    ProviderErrorNormalizer,
    Prompt,
    Response,
    hookimpl,
)
import llm
from llm.parts import StreamEvent

from abc import abstractmethod
from typing import (
    Any,
    AsyncGenerator,
    cast,
    Dict,
    List,
    Iterator,
    Optional,
    Union,
)
import json


class BaseProviderAdapter(ProviderAdapter):
    """
    Provider 适配器基类，简化常见操作的实现。

    外部插件可以继承此类来快速实现 provider 适配器，
    只需要重写抽象方法即可。
    """

    def __init__(self, model_instance: Any):
        self.model_instance = model_instance

    def build_messages(
        self, prompt: Prompt, *, conversation=None, image_detail=None
    ) -> List[Dict[str, Any]]:
        if hasattr(self.model_instance, "build_messages"):
            return self.model_instance.build_messages(
                prompt, conversation, image_detail=image_detail
            )
        return self._default_build_messages(prompt, conversation)

    def _default_build_messages(
        self, prompt: Prompt, conversation=None
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if conversation is not None:
            for prev_response in conversation.responses:
                messages.append({"role": "user", "content": prev_response.prompt.prompt})
                messages.append(
                    {"role": "assistant", "content": cast(Response, prev_response).text()}
                )
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})
        messages.append({"role": "user", "content": prompt.prompt})
        return messages

    def build_request_kwargs(self, prompt: Prompt, stream: bool) -> Dict[str, Any]:
        if hasattr(self.model_instance, "build_kwargs"):
            return self.model_instance.build_kwargs(prompt, stream)
        return {}

    def get_client(self, key: Optional[str], *, async_: bool = False) -> Any:
        if hasattr(self.model_instance, "get_client"):
            return self.model_instance.get_client(key, async_=async_)
        return None

    @abstractmethod
    def process_streaming_response(
        self, response_object, response: Response
    ) -> Iterator[Union[str, StreamEvent]]:
        pass

    @abstractmethod
    def process_non_streaming_response(
        self, response_object, response: Response
    ) -> Iterator[Union[str, StreamEvent]]:
        pass

    @abstractmethod
    def normalize_error(self, exception: Exception) -> Exception:
        pass


class AnthropicProviderAdapter(BaseProviderAdapter):
    """
    Anthropic (Claude) Provider 适配器示例实现。

    外部插件（如 llm-anthropic）可以直接使用或继承此类来实现统一的接口。
    """

    provider_name = "anthropic"

    def normalize_error(self, exception: Exception) -> Exception:
        return ProviderErrorNormalizer.normalize_anthropic_error(exception)

    def process_streaming_response(
        self, response_object, response: Response
    ) -> Iterator[Union[str, StreamEvent]]:
        full_text = ""
        tool_calls: Dict[str, Dict[str, Any]] = {}
        usage: Dict[str, Any] = {}

        for event in response_object:
            event_type = getattr(event, "type", None)

            if event_type == "content_block_delta":
                delta = getattr(event, "delta", None)
                if delta:
                    delta_type = getattr(delta, "type", None)

                    if delta_type == "text_delta":
                        text = getattr(delta, "text", "")
                        if text:
                            full_text += text
                            yield StreamEvent(type="text", chunk=text)

                    elif delta_type == "input_json_delta":
                        partial_json = getattr(delta, "partial_json", "")
                        if partial_json:
                            idx = getattr(event, "index", 0)
                            if idx not in tool_calls:
                                tool_calls[idx] = {
                                    "id": None,
                                    "name": None,
                                    "arguments": "",
                                }
                            tool_calls[idx]["arguments"] += partial_json
                            if tool_calls[idx]["name"]:
                                yield StreamEvent(
                                    type="tool_call_args",
                                    chunk=partial_json,
                                    tool_call_id=tool_calls[idx]["id"],
                                )

            elif event_type == "content_block_start":
                content_block = getattr(event, "content_block", None)
                if content_block:
                    block_type = getattr(content_block, "type", None)

                    if block_type == "tool_use":
                        idx = getattr(event, "index", 0)
                        tool_calls[idx] = {
                            "id": getattr(content_block, "id", None),
                            "name": getattr(content_block, "name", None),
                            "arguments": "",
                        }
                        if tool_calls[idx]["name"]:
                            yield StreamEvent(
                                type="tool_call_name",
                                chunk=tool_calls[idx]["name"],
                                tool_call_id=tool_calls[idx]["id"],
                            )

            elif event_type == "message_delta":
                delta = getattr(event, "delta", None)
                if delta:
                    stop_reason = getattr(delta, "stop_reason", None)
                    if stop_reason == "max_tokens":
                        pass

            elif event_type == "message_stop":
                message = getattr(event, "message", None)
                if message:
                    usage_obj = getattr(message, "usage", None)
                    if usage_obj:
                        usage = {
                            "prompt_tokens": getattr(usage_obj, "input_tokens", None),
                            "completion_tokens": getattr(
                                usage_obj, "output_tokens", None
                            ),
                        }

        self._finalize_response(response, full_text, tool_calls, usage)

    def process_non_streaming_response(
        self, response_object, response: Response
    ) -> Iterator[Union[str, StreamEvent]]:
        full_text = ""
        tool_calls: Dict[str, Dict[str, Any]] = {}
        usage: Dict[str, Any] = {}

        content = getattr(response_object, "content", [])
        for block in content:
            block_type = getattr(block, "type", None)

            if block_type == "text":
                text = getattr(block, "text", "")
                if text:
                    full_text += text
                    yield StreamEvent(type="text", chunk=text)

            elif block_type == "tool_use":
                idx = getattr(block, "id", str(len(tool_calls)))
                tool_calls[idx] = {
                    "id": idx,
                    "name": getattr(block, "name", None),
                    "arguments": getattr(block, "input", {}),
                }
                if tool_calls[idx]["name"]:
                    yield StreamEvent(
                        type="tool_call_name",
                        chunk=tool_calls[idx]["name"],
                        tool_call_id=idx,
                    )
                    args_json = json.dumps(tool_calls[idx]["arguments"])
                    yield StreamEvent(
                        type="tool_call_args",
                        chunk=args_json,
                        tool_call_id=idx,
                    )

        usage_obj = getattr(response_object, "usage", None)
        if usage_obj:
            usage = {
                "prompt_tokens": getattr(usage_obj, "input_tokens", None),
                "completion_tokens": getattr(usage_obj, "output_tokens", None),
            }

        response.response_json = {
            "content": full_text,
            "usage": usage,
        }
        self._finalize_response(response, full_text, tool_calls, usage)

    def _finalize_response(
        self,
        response: Response,
        full_text: str,
        tool_calls: Dict[str, Dict[str, Any]],
        usage: Dict[str, Any],
    ):
        if tool_calls:
            for tool_call in tool_calls.values():
                if tool_call["name"] is not None:
                    try:
                        arguments = tool_call["arguments"]
                        if isinstance(arguments, str):
                            if arguments:
                                arguments = json.loads(arguments)
                            else:
                                arguments = {}
                        response.add_tool_call(
                            llm.ToolCall(
                                tool_call_id=tool_call["id"],
                                name=tool_call["name"],
                                arguments=arguments,
                            )
                        )
                    except (json.JSONDecodeError, TypeError):
                        pass
        self.set_usage(response, usage)


class GeminiProviderAdapter(BaseProviderAdapter):
    """
    Google Gemini Provider 适配器示例实现。

    外部插件（如 llm-gemini）可以直接使用或继承此类来实现统一的接口。
    """

    provider_name = "gemini"

    def normalize_error(self, exception: Exception) -> Exception:
        return ProviderErrorNormalizer.normalize_gemini_error(exception)

    def process_streaming_response(
        self, response_object, response: Response
    ) -> Iterator[Union[str, StreamEvent]]:
        full_text = ""
        tool_calls: Dict[int, Dict[str, Any]] = {}
        usage: Dict[str, Any] = {}

        for chunk_idx, chunk in enumerate(response_object):
            candidates = getattr(chunk, "candidates", [])
            if not candidates:
                continue

            candidate = candidates[0]
            content = getattr(candidate, "content", None)
            if not content:
                continue

            parts = getattr(content, "parts", [])
            for part_idx, part in enumerate(parts):
                text = getattr(part, "text", None)
                if text:
                    full_text += text
                    yield StreamEvent(type="text", chunk=text)

                function_call = getattr(part, "function_call", None)
                if function_call:
                    tool_name = getattr(function_call, "name", None)
                    args = getattr(function_call, "args", {})

                    if tool_name:
                        existing_idx = None
                        for idx, tc in tool_calls.items():
                            if tc["name"] == tool_name:
                                existing_idx = idx
                                break

                        if existing_idx is None:
                            new_idx = len(tool_calls)
                            tool_calls[new_idx] = {
                                "id": None,
                                "name": tool_name,
                                "arguments": {},
                            }
                            yield StreamEvent(
                                type="tool_call_name",
                                chunk=tool_name,
                                tool_call_id=None,
                            )

                        idx = existing_idx if existing_idx is not None else new_idx

                        if args:
                            tool_calls[idx]["arguments"].update(args)
                            yield StreamEvent(
                                type="tool_call_args",
                                chunk=json.dumps(args),
                                tool_call_id=None,
                            )

            usage_metadata = getattr(candidate, "usage_metadata", None)
            if usage_metadata:
                usage = {
                    "prompt_tokens": getattr(usage_metadata, "total_token_count", None),
                    "completion_tokens": None,
                }

        self._finalize_response(response, full_text, tool_calls, usage)

    def process_non_streaming_response(
        self, response_object, response: Response
    ) -> Iterator[Union[str, StreamEvent]]:
        full_text = ""
        tool_calls: Dict[int, Dict[str, Any]] = {}
        usage: Dict[str, Any] = {}

        candidates = getattr(response_object, "candidates", [])
        if not candidates:
            return

        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        if not content:
            return

        parts = getattr(content, "parts", [])
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                full_text += text
                yield StreamEvent(type="text", chunk=text)

            function_call = getattr(part, "function_call", None)
            if function_call:
                idx = len(tool_calls)
                tool_name = getattr(function_call, "name", None)
                args = getattr(function_call, "args", {})
                if tool_name:
                    tool_calls[idx] = {
                        "id": None,
                        "name": tool_name,
                        "arguments": args,
                    }
                    yield StreamEvent(
                        type="tool_call_name",
                        chunk=tool_name,
                        tool_call_id=None,
                    )
                    yield StreamEvent(
                        type="tool_call_args",
                        chunk=json.dumps(args),
                        tool_call_id=None,
                    )

        usage_metadata = getattr(candidate, "usage_metadata", None)
        if usage_metadata:
            usage = {
                "prompt_tokens": getattr(usage_metadata, "total_token_count", None),
                "completion_tokens": None,
            }

        response.response_json = {
            "content": full_text,
            "usage": usage,
        }
        self._finalize_response(response, full_text, tool_calls, usage)

    def _finalize_response(
        self,
        response: Response,
        full_text: str,
        tool_calls: Dict[int, Dict[str, Any]],
        usage: Dict[str, Any],
    ):
        if tool_calls:
            for tool_call in tool_calls.values():
                if tool_call["name"] is not None:
                    try:
                        response.add_tool_call(
                            llm.ToolCall(
                                tool_call_id=tool_call["id"],
                                name=tool_call["name"],
                                arguments=tool_call["arguments"],
                            )
                        )
                    except (json.JSONDecodeError, TypeError):
                        pass
        self.set_usage(response, usage)
