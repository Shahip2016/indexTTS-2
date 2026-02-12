import argparse
import os
from indextts2.inference import IndexTTS2

def main():
    parser = argparse.ArgumentParser(description="IndexTTS-2 Inference CLI")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--voice", type=str, required=True, help="Path to reference speaker audio")
    parser.add_argument("--output", type=str, default="output.wav", help="Path to save generated audio")
    parser.add_argument("--model_dir", type=str, default="checkpoints", help="Directory containing model checkpoints")
    parser.add_argument("--config", type=str, default="checkpoints/config.yaml", help="Path to config file")
    parser.add_argument("--emotion", type=str, default=None, help="Optional text description of desired emotion")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.voice):
        print(f"Error: Voice file '{args.voice}' not found.")
        return

    try:
        tts = IndexTTS2(cfg_path=args.config, model_dir=args.model_dir)
        tts.infer(text=args.text, spk_audio_path=args.voice, output_path=args.output, emo_description=args.emotion)
        print("Done.")
    except Exception as e:
        print(f"An error occurred during inference: {e}")

if __name__ == "__main__":
    main()
