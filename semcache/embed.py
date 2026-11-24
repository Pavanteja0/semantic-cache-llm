"""Offline embedder: hashed word + character-trigram vectors, L2-normalised.
Any function str -> list[float] can replace it (e.g. a real embedding API)."""
import hashlib
import math
import re

DIM = 512
STOP = {"the", "a", "an", "is", "are", "to", "of", "do", "does", "i", "you", "me", "my", "can", "please", "how", "what"}


def _h(tok):
    return int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM


def embed(text):
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]
    v = [0.0] * DIM
    for w in words:
        v[_h("w:" + w)] += 2.0
        for i in range(len(w) - 2):
            v[_h("c:" + w[i:i + 3])] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))
