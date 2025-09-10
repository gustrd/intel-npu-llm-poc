import subprocess
import os
import sys

def main():
    # Read the first 2000 characters from examples/story.txt
    with open('examples/story.txt', 'r', encoding='utf-8') as f:
        text = f.read(2000)
    
    # Construct the prompt for summarization
    prompt = f"Please provide a concise summary of the following text:\n\n{text}"
    
    # Get the absolute path to the main.py script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    main_script = os.path.join(parent_dir, 'src', 'main.py')
    
    # Command to run main.py with necessary arguments
    cmd = [
        sys.executable, main_script,
        '--prompt', prompt,
        '--save-directory', os.path.join(parent_dir, 'model_weights'),
        '--n-predict', '512',
        '--max-context-len', '8192',
        '--max-prompt-len', '8192',
        '--disable-streaming'
    ]
    
    # Run the command and capture output, suppressing warnings by redirecting stderr
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print the output from inference.py
    print(result.stdout)
    # Only print stderr if there was a real error (non-zero exit code)
    if result.returncode != 0 and result.stderr:
        print("Error:", result.stderr)

if __name__ == '__main__':
    main()