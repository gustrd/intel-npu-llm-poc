
import sys
from unittest.mock import MagicMock, patch
import argparse
import unittest

# Mock external dependencies before importing main
sys.modules['torch'] = MagicMock()
sys.modules['ipex_llm'] = MagicMock()
sys.modules['ipex_llm.transformers'] = MagicMock()
sys.modules['ipex_llm.transformers.npu_model'] = MagicMock()
sys.modules['transformers'] = MagicMock()

from src.main import main

class TestMain(unittest.TestCase):
    @patch('src.main.argparse.ArgumentParser.parse_args')
    @patch('src.main.NPUInferenceEngine')
    def test_main_execution(self, mock_engine_cls, mock_parse_args):
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
            save_directory="./model_weights"
        )
        mock_parse_args.return_value = args
        
        # Setup mock engine instance
        mock_engine_instance = MagicMock()
        mock_engine_cls.return_value = mock_engine_instance
        
        mock_engine_instance.generate.return_value = {
            "input_tokens": 10,
            "generated_tokens": 5,
            "generation_time": 1.0,
            "tokens_per_second": 5.0,
            "output_text": "Output"
        }
        
        main()
        
        # Verify engine initialization
        mock_engine_cls.assert_called_once_with(
            model_path='mock/model',
            save_directory='./model_weights',
            low_bit='sym_int4',
            max_context_len=512,
            max_prompt_len=256,
            quantization_group_size=0
        )
        
        # Verify engine calls
        mock_engine_instance.load_model.assert_called_once()
        mock_engine_instance.generate.assert_called_once_with(
            prompt="Test prompt",
            n_predict=50,
            stream=False
        )

if __name__ == '__main__':
    unittest.main()