
import unittest
from unittest.mock import MagicMock, patch
import sys

# Mock external dependencies
sys.modules['torch'] = MagicMock()
sys.modules['ipex_llm'] = MagicMock()
sys.modules['ipex_llm.transformers'] = MagicMock()
sys.modules['ipex_llm.transformers.npu_model'] = MagicMock()
sys.modules['transformers'] = MagicMock()

from src.engine import NPUInferenceEngine

class TestNPUInferenceEngine(unittest.TestCase):
    def setUp(self):
        self.model_path = "mock/model"
        self.save_directory = "./mock_weights"
        self.engine = NPUInferenceEngine(
            model_path=self.model_path,
            save_directory=self.save_directory
        )

    @patch('src.engine.os.path.exists')
    @patch('src.engine.AutoModelForCausalLM')
    @patch('src.engine.AutoTokenizer')
    def test_load_model_not_exists(self, mock_tokenizer, mock_model, mock_exists):
        mock_exists.return_value = False
        
        # Setup mocks
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        self.engine.load_model()
        
        # Verify calls
        mock_model.from_pretrained.assert_called_once()
        mock_tokenizer.from_pretrained.assert_called_once()
        mock_tokenizer_instance.save_pretrained.assert_called_once_with(self.save_directory)

    @patch('src.engine.os.path.exists')
    @patch('src.engine.AutoModelForCausalLM')
    @patch('src.engine.AutoTokenizer')
    def test_load_model_exists(self, mock_tokenizer, mock_model, mock_exists):
        mock_exists.return_value = True
        
        self.engine.load_model()
        
        # Verify calls
        mock_model.load_low_bit.assert_called_once()
        mock_tokenizer.from_pretrained.assert_called_once_with(self.save_directory, trust_remote_code=True)

    def test_generate_without_load(self):
        with self.assertRaises(RuntimeError):
            self.engine.generate("test")

    @patch('src.engine.time.time')
    def test_generate(self, mock_time):
        # Setup engine state
        self.engine.model = MagicMock()
        self.engine.tokenizer = MagicMock()
        
        # Setup mocks
        self.engine.tokenizer.apply_chat_template.return_value = "formatted prompt"
        self.engine.tokenizer.return_value.input_ids = [[1, 2, 3]]
        self.engine.tokenizer.decode.return_value = "generated text"
        
        mock_output = [[1, 2, 3, 4, 5]]
        self.engine.model.generate.return_value = mock_output
        
        mock_time.side_effect = [100.0, 101.0, 102.0, 103.0, 104.0] # Start, End, and extras
        
        metrics = self.engine.generate("test prompt", stream=False)
        
        self.assertEqual(metrics['input_tokens'], 3)
        self.assertEqual(metrics['generated_tokens'], 2) # 5 - 3
        self.assertAlmostEqual(metrics['generation_time'], 1.0)
        self.assertEqual(metrics['output_text'], "generated text")

if __name__ == '__main__':
    unittest.main()
