import pytest
import llm
import llm.cli
from click.testing import CliRunner
from unittest.mock import patch, MagicMock, ANY
import sys
from llm.parts import Message, TextPart


class TestThresholdTrigger:
    """Tests for compression threshold triggering"""

    def test_calculate_total_tokens(self, mock_model):
        """Test that total tokens are calculated correctly from responses."""
        conversation = llm.Conversation(model=mock_model)
        
        mock_response1 = MagicMock()
        mock_response1.input_tokens = 100
        mock_response1.output_tokens = 50
        
        mock_response2 = MagicMock()
        mock_response2.input_tokens = 150
        mock_response2.output_tokens = 75
        
        conversation.responses = [mock_response1, mock_response2]
        
        assert conversation._calculate_total_tokens() == 375

    def test_should_compress_disabled(self, mock_model):
        """Test that compression is NOT triggered when disabled."""
        conversation = llm.Conversation(model=mock_model)
        conversation.compress_enabled = False
        conversation.compress_threshold = 100
        
        mock_response = MagicMock()
        mock_response.input_tokens = 100
        mock_response.output_tokens = 100
        conversation.responses = [mock_response]
        
        assert conversation._should_compress() == False

    def test_should_compress_no_threshold(self, mock_model):
        """Test that compression is NOT triggered when threshold is None."""
        conversation = llm.Conversation(model=mock_model)
        conversation.compress_enabled = True
        conversation.compress_threshold = None
        
        mock_response = MagicMock()
        mock_response.input_tokens = 1000
        mock_response.output_tokens = 1000
        conversation.responses = [mock_response]
        
        assert conversation._should_compress() == False

    def test_should_compress_below_threshold(self, mock_model):
        """Test that compression is NOT triggered when below threshold."""
        conversation = llm.Conversation(model=mock_model)
        conversation.compress_enabled = True
        conversation.compress_threshold = 200
        
        mock_response = MagicMock()
        mock_response.input_tokens = 50
        mock_response.output_tokens = 50
        conversation.responses = [mock_response]
        
        assert conversation._should_compress() == False

    def test_should_compress_at_threshold(self, mock_model):
        """Test that compression IS triggered when at threshold."""
        conversation = llm.Conversation(model=mock_model)
        conversation.compress_enabled = True
        conversation.compress_threshold = 200
        
        mock_response = MagicMock()
        mock_response.input_tokens = 100
        mock_response.output_tokens = 100
        conversation.responses = [mock_response]
        
        assert conversation._should_compress() == True

    def test_should_compress_above_threshold(self, mock_model):
        """Test that compression IS triggered when above threshold."""
        conversation = llm.Conversation(model=mock_model)
        conversation.compress_enabled = True
        conversation.compress_threshold = 100
        
        mock_response1 = MagicMock()
        mock_response1.input_tokens = 100
        mock_response1.output_tokens = 50
        
        mock_response2 = MagicMock()
        mock_response2.input_tokens = 100
        mock_response2.output_tokens = 50
        
        conversation.responses = [mock_response1, mock_response2]
        
        assert conversation._should_compress() == True


class TestSummaryPromptGeneration:
    """Tests for summary prompt generation"""

    def test_generate_summary_prompt_basic(self, mock_model):
        """Test that summary prompt includes conversation history."""
        conversation = llm.Conversation(model=mock_model)
        
        messages = [
            Message(role="user", parts=[TextPart(text="Hello")]),
            Message(role="assistant", parts=[TextPart(text="Hi there!")]),
        ]
        
        prompt = conversation._generate_summary_prompt(messages)
        
        assert "user: Hello" in prompt
        assert "assistant: Hi there!" in prompt
        assert "请总结以下对话的关键要点" in prompt

    def test_split_system_and_other_messages(self, mock_model):
        """Test that system messages are separated from other messages."""
        conversation = llm.Conversation(model=mock_model)
        
        system_msg = Message(role="system", parts=[TextPart(text="System prompt")])
        user_msg = Message(role="user", parts=[TextPart(text="Hello")])
        assistant_msg = Message(role="assistant", parts=[TextPart(text="Hi")])
        
        messages = [system_msg, user_msg, assistant_msg]
        
        system, others = conversation._split_system_and_other_messages(messages)
        
        assert system == system_msg
        assert others == [user_msg, assistant_msg]

    def test_split_system_and_other_messages_no_system(self, mock_model):
        """Test that splitting works without system message."""
        conversation = llm.Conversation(model=mock_model)
        
        user_msg = Message(role="user", parts=[TextPart(text="Hello")])
        assistant_msg = Message(role="assistant", parts=[TextPart(text="Hi")])
        
        messages = [user_msg, assistant_msg]
        
        system, others = conversation._split_system_and_other_messages(messages)
        
        assert system is None
        assert others == [user_msg, assistant_msg]


class TestBuildCompressedChain:
    """Tests for building compressed message chains"""

    def test_build_compressed_chain_with_system(self, mock_model):
        """Test building compressed chain with system message."""
        conversation = llm.Conversation(model=mock_model)
        
        system_msg = Message(role="system", parts=[TextPart(text="System prompt")])
        summary_text = "This is a summary of the conversation"
        new_user_msg = "What's next?"
        
        compressed = conversation._build_compressed_chain(
            system_msg, summary_text, new_user_msg
        )
        
        assert len(compressed) == 3
        assert compressed[0] == system_msg
        assert compressed[1].role == "system"
        assert "对话历史摘要：" in compressed[1].parts[0].text
        assert summary_text in compressed[1].parts[0].text
        assert compressed[2].role == "user"
        assert compressed[2].parts[0].text == new_user_msg

    def test_build_compressed_chain_without_system(self, mock_model):
        """Test building compressed chain without system message."""
        conversation = llm.Conversation(model=mock_model)
        
        summary_text = "This is a summary"
        new_user_msg = "What's next?"
        
        compressed = conversation._build_compressed_chain(
            None, summary_text, new_user_msg
        )
        
        assert len(compressed) == 2
        assert compressed[0].role == "system"
        assert "对话历史摘要：" in compressed[0].parts[0].text
        assert compressed[1].role == "user"

    def test_build_compressed_chain_without_new_message(self, mock_model):
        """Test building compressed chain without new user message."""
        conversation = llm.Conversation(model=mock_model)
        
        summary_text = "This is a summary"
        
        compressed = conversation._build_compressed_chain(
            None, summary_text, None
        )
        
        assert len(compressed) == 1
        assert compressed[0].role == "system"


class TestSummaryModelSelection:
    """Tests for summary model selection"""

    def test_get_summary_model_uses_current_by_default(self, mock_model):
        """Test that current model is used when no compress_model_id specified."""
        conversation = llm.Conversation(model=mock_model)
        conversation.compress_model_id = None
        
        with patch('llm.get_model', side_effect=Exception("Model not found")):
            summary_model = conversation._get_summary_model()
            assert summary_model == mock_model

    def test_get_summary_model_uses_specified_model(self, mock_model):
        """Test that specified model is used when available."""
        conversation = llm.Conversation(model=mock_model)
        conversation.compress_model_id = "summary-model"
        
        mock_summary_model = MagicMock()
        mock_summary_model.model_id = "summary-model"
        
        with patch('llm.get_model', return_value=mock_summary_model) as mock_get:
            summary_model = conversation._get_summary_model()
            assert summary_model == mock_summary_model
            mock_get.assert_called_once_with("summary-model")

    def test_get_summary_model_falls_back_on_error(self, mock_model):
        """Test that it falls back to current model when specified model not found."""
        conversation = llm.Conversation(model=mock_model)
        conversation.compress_model_id = "non-existent-model"
        
        with patch('llm.get_model', side_effect=ValueError("Model not found")):
            summary_model = conversation._get_summary_model()
            assert summary_model == mock_model


class TestCompressHistory:
    """Tests for the main _compress_history method"""

    def test_compress_history_no_history(self, mock_model):
        """Test that compression returns None when there's no history."""
        conversation = llm.Conversation(model=mock_model)
        conversation.compress_enabled = True
        conversation.compress_threshold = 10
        
        conversation.responses = []
        
        result = conversation._compress_history("new message")
        assert result is None

    def test_compress_history_not_triggered(self, mock_model):
        """Test that compression returns None when threshold not exceeded."""
        conversation = llm.Conversation(model=mock_model)
        conversation.compress_enabled = True
        conversation.compress_threshold = 1000
        
        mock_response = MagicMock()
        mock_response.input_tokens = 10
        mock_response.output_tokens = 10
        mock_response.prompt = MagicMock()
        mock_response.prompt.messages = []
        mock_response._messages_now.return_value = []
        conversation.responses = [mock_response]
        
        result = conversation._compress_history("new message")
        assert result is None

    def test_compress_history_success(self, mock_model):
        """Test successful compression of history."""
        conversation = llm.Conversation(model=mock_model)
        conversation.compress_enabled = True
        conversation.compress_threshold = 10
        conversation.compress_model_id = None
        
        mock_prompt = MagicMock()
        mock_prompt.messages = [
            Message(role="user", parts=[TextPart(text="Hello")]),
            Message(role="assistant", parts=[TextPart(text="Hi there!")]),
        ]
        
        mock_response = MagicMock()
        mock_response.input_tokens = 100
        mock_response.output_tokens = 100
        mock_response.prompt = mock_prompt
        mock_response._messages_now.return_value = []
        conversation.responses = [mock_response]
        
        mock_summary_model = MagicMock()
        mock_summary_response = MagicMock()
        mock_summary_response.text.return_value = "User said hello, assistant replied"
        mock_summary_model.prompt.return_value = mock_summary_response
        
        with patch.object(conversation, '_get_summary_model', return_value=mock_summary_model):
            result = conversation._compress_history("What's next?")
            
            assert result is not None
            assert "original_token_count" in result
            assert result["original_token_count"] == 200
            assert "summary_text" in result
            assert result["summary_text"] == "User said hello, assistant replied"
            assert "summary_model_id" in result
            assert "compressed_messages" in result
            assert len(result["compressed_messages"]) >= 2


class TestCompressionLogging:
    """Tests for compression event logging to database"""

    def test_compressions_table_exists_in_schema(self, logs_db):
        """Test that the compressions table is created in the database schema."""
        import llm.migrations
        from llm.migrations import migrate
        
        migrate(logs_db)
        
        assert "compressions" in logs_db.table_names()

    def test_compressions_table_has_correct_columns(self, logs_db):
        """Test that the compressions table has all required columns."""
        import llm.migrations
        from llm.migrations import migrate
        
        migrate(logs_db)
        
        columns = [c.name for c in logs_db["compressions"].columns]
        expected_columns = [
            "id",
            "conversation_id",
            "response_id",
            "original_token_count",
            "summary_text",
            "summary_model_id",
            "datetime_utc",
        ]
        
        for col in expected_columns:
            assert col in columns, f"Missing column: {col}"

    def test_log_to_db_records_compression(self, mock_model, logs_db):
        """Test that log_to_db records compression info when available."""
        import llm.migrations
        from llm.migrations import migrate
        
        migrate(logs_db)
        
        conversation = llm.Conversation(model=mock_model)
        conversation.compress_enabled = True
        conversation.compress_threshold = 100
        
        mock_model.enqueue(["First response"])
        response1 = conversation.prompt("Hello", stream=False)
        response1._compression_info = {
            "original_token_count": 100,
            "summary_text": "Test summary",
            "summary_model_id": "mock",
        }
        response1.log_to_db(logs_db)
        
        compressions = list(logs_db["compressions"].rows)
        assert len(compressions) == 1
        assert compressions[0]["original_token_count"] == 100
        assert compressions[0]["summary_text"] == "Test summary"
        assert compressions[0]["summary_model_id"] == "mock"
        assert compressions[0]["conversation_id"] == conversation.id
        assert compressions[0]["response_id"] == response1.id

    def test_log_to_db_no_compression_info(self, mock_model, logs_db):
        """Test that log_to_db works normally when no compression info."""
        import llm.migrations
        from llm.migrations import migrate
        
        migrate(logs_db)
        
        conversation = llm.Conversation(model=mock_model)
        
        mock_model.enqueue(["Response"])
        response = conversation.prompt("Hello", stream=False)
        response.log_to_db(logs_db)
        
        compressions = list(logs_db["compressions"].rows)
        assert len(compressions) == 0


@pytest.mark.xfail(sys.platform == "win32", reason="Expected to fail on Windows")
class TestChatCompressionCLI:
    """Tests for CLI compression options"""

    def test_chat_with_compress_flag(self, mock_model, logs_db):
        """Test that --compress flag works in chat mode."""
        runner = CliRunner()
        mock_model.enqueue(["Response 1"])
        mock_model.enqueue(["Response 2"])
        
        result = runner.invoke(
            llm.cli.cli,
            ["chat", "-m", "mock", "--compress", "--compress-threshold", "10"],
            input="Hi\nHello again\nquit\n",
            catch_exceptions=False,
        )
        
        assert result.exit_code == 0
        assert "Chatting with mock" in result.output
        assert "Response 1" in result.output
        assert "Response 2" in result.output

    def test_chat_with_compress_model_option(self, mock_model, logs_db):
        """Test that --compress-model option works."""
        runner = CliRunner()
        mock_model.enqueue(["Response 1"])
        
        result = runner.invoke(
            llm.cli.cli,
            [
                "chat",
                "-m",
                "mock",
                "--compress",
                "--compress-model",
                "mock",
                "--compress-threshold",
                "100",
            ],
            input="Hi\nquit\n",
            catch_exceptions=False,
        )
        
        assert result.exit_code == 0
        assert "Chatting with mock" in result.output
        assert "Response 1" in result.output

    def test_chat_default_compress_threshold(self, mock_model, logs_db):
        """Test that default compress threshold is applied when --compress is set."""
        runner = CliRunner()
        mock_model.enqueue(["Response"])
        
        result = runner.invoke(
            llm.cli.cli,
            ["chat", "-m", "mock", "--compress"],
            input="Hi\nquit\n",
            catch_exceptions=False,
        )
        
        assert result.exit_code == 0
        assert "Chatting with mock" in result.output
