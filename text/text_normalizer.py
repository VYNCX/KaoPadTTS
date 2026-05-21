import re
from pythainlp.util import num_to_thaiword, normalize, maiyamok
from ssg import syllable_tokenize
from eng2tha import transliterator 

def chunk_words(text, max_chars=150):
    sylls = syllable_tokenize(text)
    chunks = []
    current = ""
    for syll in sylls:
        if not current:
            current = syll
        elif len(current + syll) <= max_chars:
            current += syll
        else:
            chunks.append(current)
            current = syll
    if current:
        chunks.append(current)
    return chunks

def split_text_whitespace(text, max_chars=150):
    words = text.split()

    if len(words) == 1 and len(text) > max_chars:
        return chunk_words(text, max_chars=max_chars)

    chunks = []
    current = ""

    for word in words:
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= max_chars:
            current += " " + word
        else:
            chunks.append(current)
            current = word

    if current:
        chunks.append(current)

    return chunks

def number_to_text(text):
    pattern = r"([-+]?\d*\.\d+|\d+)"
    def replacer(match):
        num_str = match.group(0)
        try:
            if '.' in num_str:
                integer_part, decimal_part = num_str.split('.')
                integer_word = num_to_thaiword(int(integer_part))
                decimal_word = ''.join([num_to_thaiword(int(d)) for d in decimal_part])
                thai_word = f"{integer_word}จุด{decimal_word}"
            else:
                thai_word = num_to_thaiword(int(num_str))
            return thai_word
        except Exception as e:
            return num_str
    return re.sub(pattern, replacer, text)

def normalize_text(text: str):
    text = text.replace(" ๆ", "ๆ").replace("%", "เปอร์เซ็น").replace(".", "").replace(",", "")
    text_number = transliterator(number_to_text(text))
    text_cleaned = maiyamok(normalize(text_number))
    cleaned = "".join(text_cleaned)
    cleaned = re.sub(r"[^\u0E00-\u0E7F\s]", "", cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned