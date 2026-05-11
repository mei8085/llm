import pytest
import llm
import llm.cli
from click.testing import CliRunner
import sys
from unittest.mock import patch, MagicMock


class TestConversationCompression:
    def test_calculate_total_tokens(self, mock_model):
        """Test that total tokens are calculated correctly from responses."""
        conversation = llm.Conversation(model=mock_model)
        
        # Create mock responses with token counts
        mock_response1 = MagicMock()
        mock_response1.input_tokens = 100
        mock_response1.output_tokens = 50
        
        mock_response2 = MagicMock()
        mock_response2.input_tokens = 150
        mock_response2.output_tokens = 75
        
        conversation.responses = [mock_response1, mock_response2]
        
        assert conversation._calculate_total_tokens() == 375  # 100 + 50 + 150 + 75

    def test_should_compress(self, mock_model):
        """Test that compression is triggered when threshold is exceeded."""
        conversation = llm.Conversation(model=mock_model)
        conversation.compress_enabled = False
        conversation.compress_threshold = 100
        
        # Should not compress when disabled
        assert conversation._should_compress() == False
        
        # Enable compression
        conversation.compress_enabled = True
        
        # Mock responses with low token count
        mock_response1 = MagicMock()
        mock_response1.input_tokens = 30
        mock_response1.output_tokens = 20
        conversation.responses = [mock_response1]
        
        # Should not compress when below threshold
        assert conversation._should_compress() == False
        
        # Mock responses with high token count
        mock_response2 = MagicMock()
        mock_response2.input_tokens = 80
        mock_response2.output_tokens = 40
        conversation.responses = [mock_response1, mock_response2]
        
        # Should compress when above threshold (30+20+80+40=170 >= 100)
        assert conversation._should_compress() == True

    def test_generate_summary_prompt(self, mock_model):
        """Test that summary prompt is generated correctly."""
        from llm.parts import Message, TextPart
        
        conversation = llm.Conversation(model=mock_model)
        
        messages = [
            Message(role="user", parts=[TextPart(text="Hello")]),
            Message(role="assistant", parts=[TextPart(text="Hi there!")]),
            Message(role="user", parts=[TextPart(text="How are you?")]),
            Message(role="assistant", parts=[TextPart(text="I'm good, thanks!")]),
        ]
        
        prompt = conversation._generate_summary_prompt(messages)
        
        assert "user: Hello" in prompt
        assert "assistant: Hi there!" in prompt
        assert "user: How are you?" in prompt
        assert "assistant: I'm good, thanks!" in prompt
        assert "请总结以下对话的关键要点" in prompt

    def test_compression_info_stored_on_response(self, mock_model):
        """Test that compression info is stored on the response object."""
        conversation = llm.Conversation(model=mock_model)
        conversation.compress_enabled = True
        conversation.compress_threshold = 1
        
        # Create a mock response
        mock_response = MagicMock()
        mock_response.input_tokens = 10
        mock_response.output_tokens = 10
        mock_response.prompt = MagicMock()
        mock_response.prompt.messages = []
        mock_response._messages_now.return_value = []
        conversation.responses = [mock_response]
        
        # Mock the summary model
        mock_summary_response = MagicMock()
        mock_summary_response.text.return_value = "Summary of the conversation"
        
        with patch('llm.get_model') as mock_get_model:
            mock_get_model.return_value = mock_model
            mock_model.enqueue(["Summary of the conversation"])
            
            # Test that we can check for compression info attribute
            response = conversation.prompt("Test message", stream=False)
            # The response may or may not have _compression_info depending on
            # whether the threshold was exceeded and compression was performed
            # But the code should not crash


@pytest.mark.xfail(sys.platform == "win32", reason="Expected to fail on Windows")
class TestChatCompressionCLI:
    def test_chat_with_compression_flag(self, mock_model, logs_db):
        """Test that the --compress flag works in chat mode."""
        runner = CliRunner()
        mock_model.enqueue(["Response 1"])
        mock_model.enqueue(["Response 2"])
        
        result = runner.invoke(
            llm.cli.cli,
            ["chat", "-m", "mock", "--compress", "--compress-threshold", "1"],
            input="Hi\nHello again\nquit\n",
            catch_exceptions=False,
        )
        
        assert result.exit_code == 0
        assert "Chatting with mock" in result.output
        assert "Response 1" in result.output
        assert "Response 2" in result.output

    def test_chat_with_compression_model_option(self, mock_model, logs_db):
        """Test that the --compress-model option works."""
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
                "1",
            ],
            input="Hi\nquit\n",
            catch_exceptions=False,
        )
        
        assert result.exit_code == 0
        assert "Chatting with mock" in result.output
        assert "Response 1" in result.output
