import os
import sys
import logging

# Add the parent directory to sys.path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine import NPUInferenceEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Read the first 2000 characters from examples/story.txt
    story_path = os.path.join(os.path.dirname(__file__), 'story.txt')
    try:
        with open(story_path, 'r', encoding='utf-8') as f:
            text = f.read(2000)
    except FileNotFoundError:
        logger.error(f"File not found: {story_path}")
        sys.exit(1)
    
    # Construct the prompt for summarization
    prompt = f"Please provide a concise summary of the following text:\n\n{text}"
    
    # Configuration
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_directory = os.path.join(base_dir, 'model_weights')
    model_path = "Qwen/Qwen2.5-0.5B-Instruct" # Default model
    
    logger.info("Starting summarization...")
    
    try:
        engine = NPUInferenceEngine(
            model_path=model_path,
            save_directory=save_directory,
            n_predict=512,
            max_context_len=8192,
            max_prompt_len=8192
        )
        
        engine.load_model()
        
        metrics = engine.generate(
            prompt=prompt,
            n_predict=512,
            stream=True
        )
        
        # The output is already streamed, but we can access the full text if needed
        # logger.info(f"Full summary: {metrics['output_text']}")
        
    except Exception as e:
        logger.error(f"Summarization failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()