# Intel NPU LLM Proof of Concept

This project demonstrates optimized inference for large language models using Intel's Neural Processing Unit (NPU).

## Features
- Quantized model inference with IPEX-LLM
- Hardware verification tests
- Performance benchmarking
- Example applications

## Setup

1. Create and activate virtual environment:
```bash
uv venv .env
.env\Scripts\activate
```

2. Install other dependencies:
```bash
uv pip install -e .
```

3. Verify NPU hardware:
```bash
python tests/hardware_test.py
```

## Usage

### Run inference:
```bash
python src/main.py --prompt "What is the flavor of water?" --save-directory ./model_weights
```

### Summarize text:
```bash
python examples/summarize.py
```

## Documentation
- [Intel IPEX-LLM NPU Examples](https://github.com/intel/ipex-llm/tree/main/python/llm/example/NPU/HF-Transformers-AutoModels/LLM)
