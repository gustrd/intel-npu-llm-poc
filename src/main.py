
import os
import torch
import time
import argparse

from transformers import AutoTokenizer, TextStreamer
from ipex_llm.transformers.npu_model import AutoModelForCausalLM

from transformers.utils import logging

logger = logging.get_logger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Predict Tokens using `generate()` API for npu model"
    )
    parser.add_argument(
        "--repo-id-or-model-path",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="The huggingface repo id for the model to be downloaded"
        ", or the path to the huggingface checkpoint folder.",
    )
    parser.add_argument('--prompt', type=str, default="What is life?",
                        help='Prompt to infer')
    parser.add_argument("--n-predict", type=int, default=128, help="Max tokens to predict.")
    parser.add_argument("--max-context-len", type=int, default=1024)
    parser.add_argument("--max-prompt-len", type=int, default=512)
    parser.add_argument("--quantization-group-size", type=int, default=0)
    parser.add_argument('--low-bit', type=str, default="sym_int4",
                        help='Low bit optimizations that will be applied to the model.')
    parser.add_argument("--disable-streaming", action="store_true", default=False)
    parser.add_argument("--save-directory", type=str,
        required=True,
        help="The path of folder to save converted model, "
             "If path not exists, lowbit model will be saved there. "
             "Else, lowbit model will be loaded.",
    )

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    if not os.path.exists(args.save_directory):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="eager",
            load_in_low_bit=args.low_bit,
            optimize_model=True,
            max_context_len=args.max_context_len,
            max_prompt_len=args.max_prompt_len,
            quantization_group_size=args.quantization_group_size,
            save_directory=args.save_directory
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.save_pretrained(args.save_directory)
    else:
        model = AutoModelForCausalLM.load_low_bit(
            args.save_directory,
            attn_implementation="eager",
            torch_dtype=torch.float16,
            optimize_model=True,
            max_context_len=args.max_context_len,
            max_prompt_len=args.max_prompt_len
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.save_directory, trust_remote_code=True)
        except OSError:
            # Fall back to loading tokenizer from original model path if save directory doesn't have tokenizer files
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if args.disable_streaming:
        streamer = None
    else:
        streamer = TextStreamer(tokenizer=tokenizer, skip_special_tokens=True)

    print("-" * 80)
    print("done")
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages,
                                         tokenize=False,
                                         add_generation_prompt=True)
    with torch.inference_mode():
        print("finish to load")
        for i in range(1):
            # Measure prompt processing time
            prompt_start = time.time()
            _input_ids = tokenizer([text], return_tensors="pt").input_ids
            prompt_end = time.time()
            prompt_time = prompt_end - prompt_start
            input_tokens = len(_input_ids[0])
            
            print("-" * 20, f"Run {i+1}", "-" * 20)
            print("-" * 20, "Input", "-" * 20)
            print(f"Input length: {input_tokens} tokens")
            print(text)
            print("-" * 20, "Output", "-" * 20)
            
            # Measure token generation time
            gen_start = time.time()
            output = model.generate(
                _input_ids, num_beams=1, do_sample=False, max_new_tokens=args.n_predict, streamer=streamer
            )
            gen_end = time.time()
            gen_time = gen_end - gen_start
            generated_tokens = len(output[0]) - input_tokens
            
            if args.disable_streaming:
                output_str = tokenizer.decode(output[0], skip_special_tokens=False)
                print(output_str)
            
            # Print timing information with tokens per second
            print("-" * 20, "Performance Metrics", "-" * 20)
            print(f"Prompt processing time: {prompt_time:.4f} s")
            if prompt_time > 0:
                print(f"Prompt processing speed: {input_tokens/prompt_time:.2f} t/s")
            else:
                print("Prompt processing speed: N/A (time too short)")
            print(f"Token generation time: {gen_time:.4f} s")
            if gen_time > 0:
                print(f"Token generation speed: {generated_tokens/gen_time:.2f} t/s")
            else:
                print("Token generation speed: N/A (time too short)")
            total_time = prompt_time + gen_time
            print(f"Total inference time: {total_time:.4f} s")
            if total_time > 0:
                print(f"Total throughput: {(input_tokens + generated_tokens)/total_time:.2f} t/s")
            else:
                print("Total throughput: N/A (time too short)")

    print("-" * 80)
    print("done")
    print("success shut down")

if __name__ == "__main__":
    main()