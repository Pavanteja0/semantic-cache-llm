"""Benchmark on a small hand-written paraphrase set.
Each group is (canonical question, paraphrases that SHOULD hit, near-misses that should NOT)."""
from .cache import SemanticCache

GROUPS = [
 ("How do I reset my password?", ["I forgot my password, how can I reset it?", "reset password steps"],
  ["How do I change my email address?"]),
 ("What is your refund policy?", ["Can you explain the refund policy?", "refund policy details"],
  ["What is your privacy policy?"]),
 ("How do I cancel my subscription?", ["I want to cancel my subscription", "cancel subscription how"],
  ["How do I upgrade my subscription?"]),
 ("Where can I download my invoice?", ["download invoice where", "how to download my invoice"],
  ["Where can I download the mobile app?"]),
 ("What are your support hours?", ["when are your support hours", "support hours please"],
  ["What are your shipping rates?"]),
 ("How do I enable two factor authentication?", ["enable two factor authentication steps", "turn on two factor authentication"],
  ["How do I disable two factor authentication?"]),
]


def run(threshold):
    calls = []
    cache = SemanticCache(lambda p: calls.append(p) or "answer:" + p, threshold=threshold)
    for canon, _, _ in GROUPS:
        cache.ask(canon)
    good = bad = 0
    for canon, paras, near in GROUPS:
        for p in paras:
            if cache.ask(p)["cached"]:
                good += 1
        for n in near:
            r = cache.ask(n)
            if r["cached"] and r["response"] == "answer:" + canon:
                bad += 1
    n_para = sum(len(g[1]) for g in GROUPS)
    n_near = sum(len(g[2]) for g in GROUPS)
    return {"threshold": threshold, "paraphrase_hit_rate": round(good / n_para, 3),
            "false_hit_rate": round(bad / n_near, 3), "n_paraphrases": n_para, "n_near_misses": n_near}


if __name__ == "__main__":
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        print(run(t))
