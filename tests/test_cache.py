import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from semcache.bench import run
from semcache.cache import SemanticCache
from semcache.embed import cosine, embed


def mk(**kw):
    calls = []
    return SemanticCache(lambda p: calls.append(p) or "A:" + p, **kw), calls


class T(unittest.TestCase):
    def test_identical_hits(self):
        c, calls = mk()
        c.ask("How do I reset my password?")
        r = c.ask("How do I reset my password?")
        self.assertTrue(r["cached"])
        self.assertEqual(len(calls), 1)

    def test_paraphrase_hits(self):
        c, calls = mk(threshold=0.7)
        c.ask("How do I reset my password?")
        self.assertTrue(c.ask("I forgot my password, how can I reset it?")["cached"])

    def test_unrelated_misses(self):
        c, calls = mk()
        c.ask("How do I reset my password?")
        self.assertFalse(c.ask("What is the capital of France?")["cached"])
        self.assertEqual(len(calls), 2)

    def test_ttl_expires(self):
        t = [0.0]
        c, calls = mk(ttl=10, clock=lambda: t[0])
        c.ask("hello world question")
        t[0] = 11
        self.assertFalse(c.ask("hello world question")["cached"])

    def test_lru_eviction(self):
        c, _ = mk(max_size=2)
        for q in ["alpha topic one", "beta subject two", "gamma matter three"]:
            c.ask(q)
        self.assertEqual(len(c.store), 2)
        self.assertFalse(c.ask("alpha topic one")["cached"])

    def test_stats(self):
        c, _ = mk(cost_per_call=0.01)
        c.ask("same question here")
        c.ask("same question here")
        s = c.stats()
        self.assertEqual((s["hits"], s["misses"], s["hit_rate"], s["est_cost_saved"]), (1, 1, 0.5, 0.01))

    def test_embed_normalised(self):
        self.assertAlmostEqual(cosine(embed("billing question"), embed("billing question")), 1.0)

    def test_threshold_tradeoff(self):
        lo, hi = run(0.6), run(0.9)
        self.assertGreater(lo["paraphrase_hit_rate"], hi["paraphrase_hit_rate"])
        self.assertGreater(lo["false_hit_rate"], hi["false_hit_rate"])


if __name__ == "__main__":
    unittest.main()
