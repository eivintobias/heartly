#!/usr/bin/env python3
"""
memory_store.py — Stage 4b Part B: the episodic retrieval store (paper §6.2).

Memories are text rows; retrieval is cosine similarity over embeddings.
Backend: sentence-transformers MiniLM if installed, else TF-IDF (sklearn).
The backend actually used is recorded in every result dict.

Run directly for the pre-registered LOCAL gate:
    python memory_store.py
Pass bar (PREREG_STAGE4B.md): top-1 retrieval 5/5 on the 20-row store
(5 taught facts + 15 distractors) against the 5 Stage-4 quiz questions.
"""
import json
import sys


# ---------------------------------------------------------------- store
class MemoryStore:
    def __init__(self):
        self.rows = []          # list[str]
        self.backend = None     # "minilm" | "tfidf"
        self._model = None
        self._vecs = None

    def add(self, text):
        self.rows.append(text)
        self._vecs = None       # invalidate

    def add_many(self, texts):
        for t in texts:
            self.add(t)

    # -- embedding backends -------------------------------------------
    def _ensure_backend(self):
        if self.backend is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            self.backend = "minilm"
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._model = TfidfVectorizer(analyzer="char_wb",
                                          ngram_range=(2, 4))
            self.backend = "tfidf"

    def _embed_rows(self):
        self._ensure_backend()
        if self.backend == "minilm":
            self._vecs = self._model.encode(self.rows, normalize_embeddings=True)
        else:
            self._vecs = self._model.fit_transform(self.rows)

    def _embed_query(self, q):
        if self.backend == "minilm":
            return self._model.encode([q], normalize_embeddings=True)
        return self._model.transform([q])

    # -- retrieval ------------------------------------------------------
    def retrieve(self, query, k=1):
        """Return [(score, text), ...] top-k by cosine similarity."""
        if self._vecs is None:
            self._embed_rows()
        qv = self._embed_query(query)
        if self.backend == "minilm":
            import numpy as np
            sims = (self._vecs @ qv.T).ravel()
        else:
            from sklearn.metrics.pairwise import cosine_similarity
            sims = cosine_similarity(self._vecs, qv).ravel()
        order = sims.argsort()[::-1][:k]
        return [(float(sims[i]), self.rows[i]) for i in order]


# ------------------------------------------------- Stage 4b fixed data
# The 5 Stage-4 facts (memories as the store would log them).
FACT_MEMORIES = [
    {"key": "dog",      "memory": "The user's dog is named Zorblax.",
     "question": "What is my dog's name?",          "gold": "zorblax"},
    {"key": "codename", "memory": "The secret project codename is Velvet Aurora.",
     "question": "What is the secret project codename?",
     "gold": "velvet aurora"},
    {"key": "number",   "memory": "The user's favorite number is 7,423.",
     "question": "What is my favorite number?",     "gold": "7,423"},
    {"key": "collect",  "memory": "The user collects miniature lighthouse replicas.",
     "question": "What do I collect?",              "gold": "lighthouse"},
    {"key": "password", "memory": "The password to the lab door is mango Tuesday.",
     "question": "What is the lab door password?",  "gold": "mango tuesday"},
]

# 15 fabricated distractor memories (same style; retrieval must beat these).
DISTRACTORS = [
    "The user's cat is named Whiskertron.",
    "The user's sister lives in Trondheim.",
    "The backup project codename is Crimson Harbor.",
    "The user's lucky color is teal.",
    "The user's car is a blue 2014 hatchback.",
    "The user's favorite dessert is cloudberry cream.",
    "The wifi password at the cabin is winter5678.",
    "The user plays the accordion on weekends.",
    "The user's neighbor is called Mr. APELAND.",
    "The user's first bicycle was red with a silver bell.",
    "The office plant is a fern named Gerald.",
    "The user's favorite hiking trail is behind the old mill.",
    "The user's coffee order is a double oat-milk cortado.",
    "The garage code is 9911.",
    "The user's childhood best friend was named Solveig.",
]


def build_store():
    store = MemoryStore()
    store.add_many([f["memory"] for f in FACT_MEMORIES])
    store.add_many(DISTRACTORS)
    return store


# ------------------------------------------------------- the local gate
def run_gate(verbose=True):
    store = build_store()
    hits = 0
    results = []
    for f in FACT_MEMORIES:
        top = store.retrieve(f["question"], k=3)
        got = top[0][1]
        hit = got == f["memory"]
        hits += int(hit)
        results.append({"key": f["key"], "question": f["question"],
                        "top1": got, "top1_score": top[0][0],
                        "hit": hit,
                        "top3": [t for _, t in top]})
        if verbose:
            mark = "OK " if hit else "MISS"
            print(f"[{mark}] {f['question']}")
            for s, t in top:
                print(f"       {s:.3f}  {t}")
    report = {"backend": store.backend, "n_rows": len(store.rows),
              "top1_hits": hits, "gate_pass": hits == 5, "results": results}
    if verbose:
        print(f"\nbackend={store.backend} | top-1 {hits}/5 | "
              f"gate {'PASS' if hits == 5 else 'FAIL'}")
    return report


if __name__ == "__main__":
    rep = run_gate()
    with open("stage4b_store_gate.json", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2, ensure_ascii=False)
    print("wrote stage4b_store_gate.json")
    sys.exit(0 if rep["gate_pass"] else 1)
