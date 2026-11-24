import streamlit as st

from semcache.bench import run
from semcache.cache import SemanticCache

st.set_page_config(page_title="Semantic Cache", layout="wide")
st.title("Semantic Cache for LLM calls")
st.caption("Reworded questions are answered from cache. Default threshold 0.7 for the demo (library default 0.8). Offline hashed-vector embedder and a fake LLM that costs 0.002 per call.")

thr = st.slider("Similarity threshold", 0.5, 0.95, 0.7, 0.05)
if "cache" not in st.session_state or st.session_state.get("thr") != thr:
    st.session_state.cache = SemanticCache(lambda p: "(LLM answer to: %s)" % p, threshold=thr)
    st.session_state.thr = thr
    st.session_state.log = []
cache = st.session_state.cache

q = st.text_input("Ask something", "How do I reset my password?")
if st.button("Send", type="primary"):
    r = cache.ask(q)
    st.session_state.log.insert(0, {"question": q, "result": "CACHE HIT" if r["cached"] else "LLM call",
                                    "similarity": r["similarity"]})
s = cache.stats()
a, b, c, d = st.columns(4)
a.metric("Hits", s["hits"]); b.metric("Misses", s["misses"])
c.metric("Hit rate", "%.0f%%" % (s["hit_rate"] * 100)); d.metric("Est. cost saved", "$%.3f" % s["est_cost_saved"])
st.table(st.session_state.log)

st.subheader("Threshold tradeoff on the built-in benchmark")
st.table([run(t) for t in (0.6, 0.7, 0.8, 0.9)])
