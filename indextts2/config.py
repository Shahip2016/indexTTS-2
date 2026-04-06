import os
from omegaconf import OmegaConf

class Config:
    def __init__(self, cfg_path=None):
        self.cfg = None
        if cfg_path and os.path.exists(cfg_path):
            self.cfg = OmegaConf.load(cfg_path)
        else:
            # Default configuration structure
            self.cfg = OmegaConf.create({
                "model_dir": "checkpoints",
                "is_fp16": True,
                "device": "cuda",
                "use_cuda_kernel": False,
                "dataset": {
                    "bpe_model": "bpe.model"
                },
                "gpt": {
                    "layers": 8,
                    "model_dim": 512,
                    "heads": 8,
                    "max_text_tokens": 120,
                    "max_mel_tokens": 250,
                    "max_conditioning_inputs": 1,
                    "mel_length_compression": 1024,
                    "number_text_tokens": 256,
                    "start_text_token": 0,
                    "stop_text_token": 1,
                    "number_mel_codes": 8194,
                    "start_mel_token": 8192,
                    "stop_mel_token": 8193,
                    "gpt_checkpoint": "gpt.pth",
                    "stop_mel_token": 8193
                },
                "s2mel": {
                     "preprocess_params": {
                        "spect_params": {
                            "n_fft": 1024,
                            "win_length": 1024,
                            "hop_length": 256,
                            "n_mels": 80,
                            "sr": 22050,
                            "fmin": 0,
                            "fmax": 8000
                        }
                    },
                     "s2mel_checkpoint": "s2mel.pth"

                },
                "vocoder": {
                    "name": "nvidia/bigvgan_v2_22khz_80band_256x"
                },
                "qwen_emo_path": "Qwen-Emotion",
                "w2v_stat": "w2v_stat.pt",
                "semantic_codec": {
                     "model_name": "amphion/MaskGCT"
                },
                "emo_matrix": "emo_matrix.pt"
            })

    def load(self, cfg_path):
        self.cfg = OmegaConf.load(cfg_path)
        return self.cfg

    def get(self, key, default=None):
        return self.cfg.get(key, default)

    def save(self, save_path):
        """Save the underlying OmegaConf object to a file."""
        OmegaConf.save(self.cfg, save_path)
