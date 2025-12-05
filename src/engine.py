import os
import time
import logging
from typing import Optional, List, Dict, Any, Union
import torch
from transformers import AutoTokenizer, TextStreamer
from ipex_llm.transformers.npu_model import AutoModelForCausalLM

logger = logging.getLogger(__name__)

class NPUInferenceEngine:
    """
    Engine for running inference on Intel NPU using IPEX-LLM.
    """
    def __init__(self, 
                 model_path: str, 
                 save_directory: str, 
                 low_bit: str = "sym_int4",
                 max_context_len: int = 1024,
                 max_prompt_len: int = 512,
                 quantization_group_size: int = 0,
                 trust_remote_code: bool = True):
        
        self.model_path = model_path
        self.save_directory = save_directory
        self.low_bit = low_bit
        self.max_context_len = max_context_len
        self.max_prompt_len = max_prompt_len
        self.quantization_group_size = quantization_group_size
        self.trust_remote_code = trust_remote_code
        
        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        """
        Loads the model and tokenizer. If the model is not already saved in the save_directory,
        it loads from the original path, optimizes it, and saves it.
        """
        if not os.path.exists(self.save_directory):
            logger.info(f"Model not found in {self.save_directory}. Loading and optimizing from {self.model_path}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                trust_remote_code=self.trust_remote_code,
                attn_implementation="eager",
                load_in_low_bit=self.low_bit,
                optimize_model=True,
                max_context_len=self.max_context_len,
                max_prompt_len=self.max_prompt_len,
                quantization_group_size=self.quantization_group_size,
                save_directory=self.save_directory
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=self.trust_remote_code
            )
            self.tokenizer.save_pretrained(self.save_directory)
            logger.info(f"Model saved to {self.save_directory}")
        else:
            logger.info(f"Loading optimized model from {self.save_directory}...")
            self.model = AutoModelForCausalLM.load_low_bit(
                self.save_directory,
                attn_implementation="eager",
                torch_dtype=torch.float16,
                optimize_model=True,
                max_context_len=self.max_context_len,
                max_prompt_len=self.max_prompt_len
            )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.save_directory, 
                    trust_remote_code=self.trust_remote_code
                )
            except OSError:
                logger.warning(f"Tokenizer not found in {self.save_directory}. Fallback to {self.model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, 
                    trust_remote_code=self.trust_remote_code
                )

    def generate(self, 
                 prompt: str, 
                 n_predict: int = 128, 
                 stream: bool = True,
                 system_prompt: str = "You are a helpful assistant.") -> Dict[str, Any]:
        """
        Generates text based on the prompt.
        
        Args:
            prompt: User input text.
            n_predict: Maximum number of new tokens to generate.
            stream: Whether to stream output to stdout.
            system_prompt: System prompt for the chat template.
            
        Returns:
            Dictionary containing the output text and performance metrics.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if stream:
            streamer = TextStreamer(tokenizer=self.tokenizer, skip_special_tokens=True)
        else:
            streamer = None

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        with torch.inference_mode():
            _input_ids = self.tokenizer([text], return_tensors="pt").input_ids
            input_tokens = len(_input_ids[0])
            
            logger.info(f"Input length: {input_tokens} tokens")
            
            gen_start = time.time()
            output = self.model.generate(
                _input_ids, 
                num_beams=1, 
                do_sample=False, 
                max_new_tokens=n_predict, 
                streamer=streamer
            )
            gen_end = time.time()
            
            gen_time = gen_end - gen_start
            generated_tokens = len(output[0]) - input_tokens
            
            output_str = self.tokenizer.decode(output[0], skip_special_tokens=False)
            
            # Extract just the response part if possible, or return full
            # For now returning full decoded string as per original logic, 
            # but usually we want just the new tokens.
            # The original code printed the full decoded string if streaming was disabled.
            
            metrics = {
                "input_tokens": input_tokens,
                "generated_tokens": generated_tokens,
                "generation_time": gen_time,
                "tokens_per_second": generated_tokens / gen_time if gen_time > 0 else 0.0,
                "output_text": output_str
            }
            
            return metrics
