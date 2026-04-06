import os
import torch
import torchaudio
import numpy as np

from indextts2.config import Config
from indextts2.models.vocoder import BigVGAN
from indextts2.models.gpt.unified_voice import UnifiedVoice
from indextts2.models.s2mel.flow_matching import CFM
from indextts2.models.speaker import CAMPPlus
from indextts2.models.semantic import SemanticModel
from indextts2.models.emotion import QwenEmotion
from indextts2.utils.text import TextTokenizer, TextNormalizer
from indextts2.utils.audio import load_audio, mel_spectrogram

class IndexTTS2:
    def __init__(self, cfg_path="checkpoints/config.yaml", model_dir="checkpoints", device=None):
        self.cfg = Config(cfg_path)
        self.model_dir = model_dir
        
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"Initializing IndexTTS2 on {self.device}...")
        
        # Load models
        self.vocoder = BigVGAN(device=self.device)
        self.vocoder.load()
        
        self.gpt = UnifiedVoice(**self.cfg.get("gpt")).to(self.device)
        # self.gpt.load_state_dict(torch.load(os.path.join(model_dir, "gpt.pth")))
        
        self.speaker_encoder = CAMPPlus().to(self.device)
        self.speaker_encoder.load_pretrained()
        
        self.semantic_model = SemanticModel(device=self.device)
        # self.semantic_model.load_stats(...)
        
        self.emotion_model = QwenEmotion(model_dir, device=self.device)
        self.emotion_model.load()
        
        self.tokenizer = TextTokenizer(os.path.join(model_dir, "bpe.model"))
        
        # S2Mel model
        # self.s2mel = ...
        
    def infer(self, text, spk_audio_path, output_path=None, emo_description=None):
        print(f"Generating speech for: '{text}'")
        
        # 1. Process Text
        text_tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(text_tokens)
        input_ids = torch.tensor([token_ids]).to(self.device)
        
        # 2. Process Speaker Audio
        spk_audio = load_audio(spk_audio_path)
        if spk_audio is None:
            print("Failed to load speaker audio.")
            return None
        spk_audio = spk_audio.to(self.device)
        
        # 3. Extract Features
        # spk_emb = self.speaker_encoder(mel_spectrogram(spk_audio, ...))
        # semantic_emb = self.semantic_model.extract(spk_audio)
        
        # 4. Emotion Control
        if emo_description:
            emo_scores, _ = self.emotion_model.inference(emo_description)
            print(f"Emotion scores: {emo_scores}")
            # emotional_vector = ...
            
        # 5. GPT Generation (Autoregressive)
        # codes = self.gpt.inference(condition, input_ids)
        
        # 6. S2Mel Generation (Flow Matching)
        # mel = self.s2mel.inference(codes, ...)
        
        # 7. Vocoder Generation
        # Placeholder Mel generation for demonstration
        fake_mel = torch.randn(1, 80, 256).to(self.device) 
        wav = self.vocoder.infer(fake_mel)
        
        # 8. Save Output
        if output_path:
            torchaudio.save(output_path, wav.cpu(), 22050)
            print(f"Saved to {output_path}")
            
        return wav
