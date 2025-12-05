
import argparse
import logging
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine import NPUInferenceEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Predict Tokens using `generate()` API for NPU model"
    )
    parser.add_argument(
        "--repo-id-or-model-path",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="The huggingface repo id for the model to be downloaded, or the path to the huggingface checkpoint folder.",
    )
    parser.add_argument('--prompt', type=str, default="What is life?", help='Prompt to infer')
    parser.add_argument("--n-predict", type=int, default=128, help="Max tokens to predict.")
    parser.add_argument("--max-context-len", type=int, default=1024)
    parser.add_argument("--max-prompt-len", type=int, default=512)
    parser.add_argument("--quantization-group-size", type=int, default=0)
    parser.add_argument('--low-bit', type=str, default="sym_int4",
                        help='Low bit optimizations that will be applied to the model.')
    parser.add_argument("--disable-streaming", action="store_true", default=False)
    parser.add_argument("--save-directory", type=str,
        default="./model_weights",
        help="The path of folder to save converted model."
    )

    args = parser.parse_args()

    try:
        engine = NPUInferenceEngine(
            model_path=args.repo_id_or_model_path,
            save_directory=args.save_directory,
            low_bit=args.low_bit,
            max_context_len=args.max_context_len,
            max_prompt_len=args.max_prompt_len,
            quantization_group_size=args.quantization_group_size
        )

        engine.load_model()

        logger.info("-" * 80)
        logger.info("Model loaded successfully")
        
        logger.info("-" * 20 + " Input " + "-" * 20)
        logger.info(f"Prompt: {args.prompt}")
        
        metrics = engine.generate(
            prompt=args.prompt,
            n_predict=args.n_predict,
            stream=not args.disable_streaming
        )

        if args.disable_streaming:
            print(metrics["output_text"])

        logger.info("-" * 20 + " Performance Metrics " + "-" * 20)
        logger.info(f"Token generation time: {metrics['generation_time']:.4f} s")
        logger.info(f"Token generation speed: {metrics['tokens_per_second']:.2f} t/s")
        logger.info("-" * 80)
        logger.info("Success shut down")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()