# Semantic Cache for LLM calls

Middleware that answers a repeated or reworded prompt from cache instead of calling the model again. Looks up the most similar stored prompt by cosine similarity and serves its answer when similarity is above a threshold.

## Features
- Similarity threshold (default 0.8), LRU eviction (`max_size`), optional TTL.
- Stats: hits, misses, hit rate, estimated cost saved (`cost_per_call` is a number you set, not a real bill).
- Pluggable `llm(prompt)` and `embed_fn(text)`.

## Honest scope
- The embedder is an offline hashed word and trigram vector, so it needs no API key. It is much weaker than a real embedding model. Swap `embed_fn` for one in real use.
- Cache lookup is a linear scan, fine for thousands of entries, not millions.
- The benchmark is a small hand-written set: 6 questions, 12 paraphrases that should hit, 6 near-misses that should not. It shows the threshold tradeoff, not real-world savings.

## Measured threshold tradeoff (`python -m semcache.bench`)
| threshold | paraphrase hit rate | false hit rate |
|---|---|---|
| 0.5 | 100% | 66.7% |
| 0.6 | 100% | 66.7% |
| 0.7 | 83.3% | 16.7% |
| 0.8 | 75.0% | 16.7% |
| 0.9 | 41.7% | 0% |

A false hit means a near-miss question (for example "enable" vs "disable" two factor authentication) got another question's answer. A pure similarity cache can't separate those well, which is why the threshold needs tuning on your own data.

## Run
    python -m unittest discover -s tests    # 8 tests
Standard library only.

## Demo and try it
- **Live app, no install:** https://pavan-semantic-cache.streamlit.app (free hosting, may take a few seconds to wake up)
- Demo video with voiceover (45 s): [demo/demo.mp4](demo/demo.mp4)
- Run the Streamlit app yourself on any machine with Python 3.10+:

      git clone https://github.com/Pavanteja0/semantic-cache-llm
      cd semantic-cache-llm
      pip install -r requirements.txt
      streamlit run app.py
