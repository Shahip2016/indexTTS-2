import re
import sentencepiece as spm
import textstat

class TextNormalizer:
    def __init__(self):
        pass

    def load(self):
        # Placeholder for loading actual normalization rules if needed
        pass

    def normalize(self, text):
        # Basic normalization for now
        return text

class TextTokenizer:
    def __init__(self, bpe_model_path, normalizer=None):
        self.sp = spm.SentencePieceProcessor()
        try:
            self.sp.load(bpe_model_path)
        except Exception:
            print(f"Warning: Could not load BPE model from {bpe_model_path}")
        self.normalizer = normalizer

    def tokenize(self, text):
        if self.normalizer:
            text = self.normalizer.normalize(text)
        return self.sp.encode(text, out_type=str)

    def convert_tokens_to_ids(self, tokens):
        return self.sp.piece_to_id(tokens)

    def split_sentences(self, tokens, max_tokens=120):
        # Simple split implementation
        sentences = []
        current_sentence = []
        for token in tokens:
            current_sentence.append(token)
            if len(current_sentence) >= max_tokens:
                sentences.append(current_sentence)
                current_sentence = []
        if current_sentence:
            sentences.append(current_sentence)
        return sentences

def get_text_syllable_num(text):
    chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]')
    number_char_pattern = re.compile(r'[0-9]')
    syllable_num = 0
    tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|[0-9]+', text)
    
    if any('\u4e00' <= char <= '\u9fff' for char in text):
        for token in tokens:
            if chinese_char_pattern.search(token) or number_char_pattern.search(token):
                syllable_num += len(token)
            else:
                syllable_num += textstat.syllable_count(token)
    else:
        syllable_num = textstat.syllable_count(text)

    return syllable_num

def get_text_tts_dur(text):
    min_speed = 3
    max_speed = 5.50

    ratio = 0.8517 if any('\u4e00' <= char <= '\u9fff' for char in text) else 1.0

    syllable_num = get_text_syllable_num(text)
    max_dur = syllable_num * ratio / max_speed
    min_dur = syllable_num * ratio / min_speed

    return max_dur, min_dur
