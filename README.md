# IndexTTS-2 Implementation

This repository contains a clean implementation of [IndexTTS-2](https://arxiv.org/abs/2502.05512), an industrial-level controllable and efficient zero-shot text-to-speech system.

## Features

- **Precise Duration Control**: Support for both explicit token specification (controllable mode) and free generation (uncontrollable mode).
- **Emotion Disentanglement**: Independent control of speaker identity and emotional expression.
- **Natural Language Emotion Prompting**: Use text descriptions to guide the emotional tone of the synthesized speech.
- **Zero-Shot Cloning**: Clone voices with just a short reference audio.

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download Pretrained Models
   - Place `config.yaml` in `checkpoints/`
   - Place `gpt.pth` inside `checkpoints/`
   - Place `campplus_cn_common.bin` inside `checkpoints/`
   - Place other required checkpoints as per `config.yaml`

## Usage

### Command Line Interface

Generate speech from text using a reference voice:

```bash
python cli.py --text "Hello, this is a test of IndexTTS-2." --voice path/to/reference_audio.wav --output generated.wav
```

With emotion control:

```bash
python cli.py --text "I am so happy today!" --voice path/to/ref.wav --emotion "Extremely happy and excited"
```

### Python API

```python
from indextts2.inference import IndexTTS2

tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints")
tts.infer(text="Hello world", spk_audio_path="ref.wav", output_path="out.wav")
```

## Structure

- `indextts2/`: Main package
  - `models/`: Model definitions (GPT, S2Mel, Vocoder, etc.)
  - `utils/`: Audio and text utilities
  - `config.py`: Configuration handling
- `cli.py`: Command-line entry point

## Acknowledgements

This implementation is based on the research paper "IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System".

We acknowledge the following open-source projects that inspired or are used in this codebase:
- [BigVGAN](https://github.com/NVIDIA/BigVGAN)
- [MaskGCT](https://github.com/open-mmlab/Amphion/tree/main/models/tts/maskgct)
- [Qwen](https://huggingface.co/Qwen)
- [W2V-BERT](https://huggingface.co/facebook/w2v-bert-2.0)

## Citation

If you use this code or the original paper in your research, please cite:

```bibtex
@article{deng2025indextts,
  title={IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System},
  author={Wei Deng, Siyi Zhou, Jingchen Shu, Jinchao Wang, Lu Wang},
  journal={arXiv preprint arXiv:2502.05512},
  year={2025}
}
```
