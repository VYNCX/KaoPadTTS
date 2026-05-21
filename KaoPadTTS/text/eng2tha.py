import re
import os

def _load_dict(path):
    d = {}
    if not os.path.exists(path): return d
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            parts = line.split(",", 1)
            if len(parts) == 2:
                d[parts[0].strip().lower()] = parts[1].strip()
    return d

def _save_entry(path, word, thai):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{word},{thai}\n")
    except OSError:
        pass

class EngToThaiTransliterator:
    def __init__(self, dict_path=None):
        if dict_path is None:
            here = os.path.dirname(os.path.abspath(__file__))
            dict_path = os.path.join(here, "./dict_transliteration.txt")
        self.dict_path = dict_path
        self._map = _load_dict(dict_path)
        self._new_in_session = set()

    def transliterate_word(self, word):
        key = word.lower()
        return self._map.get(key, word)

    def transliterate_text(self, text):
        tokens = re.findall(r"[a-zA-Z]+|[^\w\s]|\s+", text)
        res = []
        for token in tokens:
            if re.fullmatch(r"\s+", token):
                res.append(token)
            elif re.fullmatch(r"[a-zA-Z]+", token):
                res.append(self._handle_word(token))
            else:
                res.append(token)
        return "".join(res)

    def add_word(self, english, thai):
        key = english.lower().strip()
        self._map[key] = thai
        self._new_in_session.add(key)
        _save_entry(self.dict_path, key, thai)

    @property
    def vocabulary_size(self):
        return len(self._map)

    def _handle_word(self, word):
        key = word.lower()
        if key in self._map:
            return self._map[key]
        if word.isupper() and len(word) > 1:
            thai = "".join(self._map.get(c.lower(), c) for c in word)
            self._learn(key, thai)
            return thai
        return word

    def _learn(self, word, thai):
        if word in self._map:
            return
        self._map[word] = thai
        if word not in self._new_in_session:
            self._new_in_session.add(word)
            _save_entry(self.dict_path, word, thai)

_THAI_CHAR = re.compile(r"[ก-๙]")
_ENG_TOKEN = re.compile(r"[A-Za-z]+")

def transliterator(text):
    translit = EngToThaiTransliterator()
    pattern = re.compile(r"([ก-๙]+|[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*|[^\sA-Za-z0-9ก-๙]+|\s+)")
    tokens = pattern.findall(text)
    res = []
    for token in tokens:
        if re.fullmatch(r"\s+", token):
            res.append(token)
        elif _THAI_CHAR.search(token):
            res.append(token)
        elif _ENG_TOKEN.search(token):
            res.append(translit.transliterate_text(token))
        else:
            res.append(token)
    return "".join(res)