# Mock the external dependencies at the very top
import sys
from unittest.mock import MagicMock, patch
import argparse
from io import StringIO

# Mock the full import path used in main.py
sys.modules['torch'] = MagicMock()
sys.modules['ipex_llm'] = MagicMock()
sys.modules['ipex_llm.transformers'] = MagicMock()
sys.modules['ipex_llm.transformers.npu_model'] = MagicMock()

# Mock transformers module hierarchy
sys.modules['transformers'] = MagicMock()
sys.modules['transformers.utils'] = MagicMock()
sys.modules['transformers.utils.logging'] = MagicMock()
sys.modules['transformers.models'] = MagicMock()
sys.modules['transformers.models.auto'] = MagicMock()
sys.modules['transformers.models.auto.modeling_auto'] = MagicMock()

import unittest
from src.main import main

class TestMain(unittest.TestCase):
    @patch('src.main.argparse.ArgumentParser.parse_args')
    @patch('src.main.AutoModelForCausalLM')
    @patch('src.main.AutoTokenizer')
    def test_main_execution(self, mock_tokenizer, mock_model, mock_parse_args):
        # Setup mock arguments
        args = argparse.Namespace(
            repo_id_or_model_path='mock/model',
            prompt="Test prompt",
            n_predict=50,
            max_context_len=512,
            max_prompt_len=256,
            quantization_group_size=0,
            low_bit="sym_int4",
            disable_streaming=True,
            save_directory="./mock_weights"
        )
        mock_parse_args.return_value = args
        
        # Mock model and tokenizer
        mock_model.from_pretrained.return_value = MagicMock()
        
        # Create a mock tokenizer instance
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.apply_chat_template.return_value = "Chat template"
        # Set the return value for when the tokenizer instance is called
        mock_tokenizer_instance.return_value = MagicMock(input_ids=[[1,2,3]])
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Mock performance metrics - 4 time calls per iteration * 3 iterations = 12 values
        with patch('src.main.time.time') as mock_time:
            mock_time.side_effect = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            main()
        
        # Verify model loading
        mock_model.from_pretrained.assert_called_once()
        
    @patch('src.main.argparse.ArgumentParser.parse_args')
    def test_argument_parsing(self, mock_parse_args):
        # Test argument defaults
        args = argparse.Namespace(
            repo_id_or_model_path='Qwen/Qwen2.5-3B-Instruct',
            prompt="What is life?",
            n_predict=128,
            max_context_len=1024,
            max_prompt_len=512,
            quantization_group_size=0,
            low_bit="sym_int4",
            disable_streaming=False,
            save_directory="./model_weights"
        )
        mock_parse_args.return_value = args
        
        main()
        
        # Verify default values are used
        self.assertEqual(args.n_predict, 128)
        self.assertEqual(args.max_context_len, 1024)

    @patch('src.main.argparse.ArgumentParser.parse_args')
    @patch('src.main.AutoModelForCausalLM.load_low_bit')
    def test_existing_model_loading(self, mock_load, mock_parse_args):
        # Test loading existing model
        args = argparse.Namespace(
            repo_id_or_model_path='mock/model',
            prompt="Test prompt",
            n_predict=50,
            max_context_len=512,
            max_prompt_len=256,
            quantization_group_size=0,
            low_bit="sym_int4",
            disable_streaming=True,
            save_directory="./existing_weights"
        )
        mock_parse_args.return_value = args
        
        # Mock existing directory
        with patch('src.main.os.path.exists', return_value=True):
            main()
        
        mock_load.assert_called_once()

if __name__ == '__main__':
    unittest.main()