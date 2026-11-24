import time
from collections import OrderedDict

from .embed import cosine, embed


class SemanticCache:
    def __init__(self, llm, threshold=0.8, max_size=1000, ttl=None, cost_per_call=0.002,
                 embed_fn=embed, clock=time.time):
        self.llm, self.threshold, self.max_size, self.ttl = llm, threshold, max_size, ttl
        self.cost_per_call, self.embed_fn, self.clock = cost_per_call, embed_fn, clock
        self.store = OrderedDict()  # key -> (vec, prompt, response, created)
        self.hits = self.misses = 0
        self._n = 0

    def _evict_expired(self):
        if self.ttl is None:
            return
        now = self.clock()
        for k in [k for k, v in self.store.items() if now - v[3] > self.ttl]:
            del self.store[k]

    def lookup(self, prompt):
        self._evict_expired()
        q = self.embed_fn(prompt)
        best_k, best = None, 0.0
        for k, (vec, _, _, _) in self.store.items():
            s = cosine(q, vec)
            if s > best:
                best_k, best = k, s
        return q, best_k, best

    def ask(self, prompt):
        q, k, score = self.lookup(prompt)
        if k is not None and score >= self.threshold:
            self.hits += 1
            self.store.move_to_end(k)
            return {"response": self.store[k][2], "cached": True, "similarity": round(score, 3)}
        self.misses += 1
        resp = self.llm(prompt)
        self._n += 1
        self.store[self._n] = (q, prompt, resp, self.clock())
        while len(self.store) > self.max_size:
            self.store.popitem(last=False)
        return {"response": resp, "cached": False, "similarity": round(score, 3)}

    def stats(self):
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0.0,
                "est_cost_saved": round(self.hits * self.cost_per_call, 4)}
