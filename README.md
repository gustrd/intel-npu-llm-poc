# Intel NPU LLM Proof of Concept

This project demonstrates optimized inference for large language models using Intel's Neural Processing Unit (NPU) via IPEX-LLM.

## Features
- **Optimized Inference**: Leverages Intel NPU for efficient LLM inference.
- **Modular Design**: Core logic encapsulated in `NPUInferenceEngine` for easy integration.
- **Hardware Verification**: Includes tools to verify NPU availability.
- **Example Applications**: Includes a summarization example.

## Setup

1. **Create and activate virtual environment**:
   ```bash
   uv venv .env
   .env\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   uv pip install -e .
   ```

3. **Verify NPU hardware**:
   ```bash
   python tests/hardware_test.py
   ```

## Usage

### Run Inference
Run the main script to generate text. The model will be automatically downloaded, optimized, and saved to `./model_weights` by default.

```bash
python src/main.py --prompt "What is the flavor of water?"
```

**Options:**
- `--prompt`: The input prompt (default: "What is life?").
- `--repo-id-or-model-path`: Hugging Face model ID or local path (default: "Qwen/Qwen2.5-0.5B-Instruct").
- `--save-directory`: Directory to save optimized model (default: `./model_weights`).
- `--n-predict`: Max tokens to predict (default: 128).
- `--disable-streaming`: Disable streaming output.

### Summarize Text
Run the example script to summarize a text file (`examples/story.txt`).

```bash
python examples/summarize.py
```

## Documentation
- [Intel IPEX-LLM NPU Examples](https://github.com/intel/ipex-llm/tree/main/python/llm/example/NPU/HF-Transformers-AutoModels/LLM)
