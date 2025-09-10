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

2. Install IPEX-LLM with NPU support:
```bash
uv pip install --pre --upgrade ipex-llm[npu]
```

3. Install other dependencies:
```bash
uv pip install -e .
```

4. Optional: For Llama-3.2-1B-Instruct & Llama-3.2-3B-Instruct
```bash
uv pip install transformers==4.45.0 accelerate==0.33.0
```

5. Verify NPU hardware:
```bash
python tests/hardware_test.py
```

For more details, see [Intel IPEX-LLM NPU Examples](https://github.com/intel/ipex-llm/tree/main/python/llm/example/NPU/HF-Transformers-AutoModels/LLM)
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
- [Project Guidelines](docs/guidelines.md)
- [Intel IPEX-LLM Documentation](https://intel.github.io/ipex-llm/)

## Contributing
Contributions welcome! Please follow our [contribution guidelines](CONTRIBUTING.md).
