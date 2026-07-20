"""BM25 retrieval over Bengali Wikipedia passages.

Builds a sparse TF-IDF index (sklearn) and retrieves top-k passages for a query.
Used to give the judge context for closed-book rows (the Bangladesh-GK gap).
"""
import json, pickle, os, sys, re, time
import numpy as np
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer

PASS = "wiki/passages.jsonl"
IDX = "wiki/tfidf.pkl"
MAT = "wiki/tfidf_matrix.npz"

def load_passages():
    P = []
    with open(PASS) as f:
        for line in f:
            d = json.loads(line)
            P.append((d["t"], d["x"]))
    return P

def build():
    t0 = time.time()
    P = load_passages()
    print(f"{len(P)} passages loaded ({time.time()-t0:.0f}s)", flush=True)
    # index on title + text so entity names weigh in
    docs = [f"{t} {t} {x}" for t, x in P]
    v = TfidfVectorizer(analyzer="word", token_pattern=r"[^\s।,;:()\"'—\-]+",
                        min_df=2, max_features=800_000, sublinear_tf=True)
    X = v.fit_transform(docs)
    print(f"tfidf {X.shape} ({time.time()-t0:.0f}s)", flush=True)
    pickle.dump({"v": v, "titles": [t for t, _ in P]}, open(IDX, "wb"))
    save_npz(MAT, X)
    json.dump([x for _, x in P], open("wiki/texts.json", "w"), ensure_ascii=False)
    print(f"saved ({time.time()-t0:.0f}s)", flush=True)

class Retriever:
    def __init__(self):
        d = pickle.load(open(IDX, "rb"))
        self.v = d["v"]; self.titles = d["titles"]
        self.X = load_npz(MAT).tocsr()
        self.texts = json.load(open("wiki/texts.json"))
        # normalize rows for cosine
        from sklearn.preprocessing import normalize
        self.X = normalize(self.X)

    def search(self, query, k=3):
        return self.search_batch([query], k)[0]

    def search_batch(self, queries, k=3):
        """One sparse matmul for all queries — orders of magnitude faster than looping."""
        from sklearn.preprocessing import normalize
        Q = normalize(self.v.transform(queries))
        S = (Q @ self.X.T).toarray()          # [n_queries, n_passages]
        out = []
        for row in S:
            top = np.argpartition(-row, k)[:k]
            top = top[np.argsort(-row[top])]
            out.append([(float(row[i]), self.titles[i], self.texts[i]) for i in top])
        return out

if __name__ == "__main__":
    if not os.path.exists(MAT) or "--rebuild" in sys.argv:
        build()
    r = Retriever()
    from common import load_samples
    S = [x for x in load_samples() if not x["context"]]
    hit = 0
    for row in S[:15]:
        res = r.search(row["prompt_bn"], k=3)
        joined = " ".join(t for _, _, t in res)
        # crude recall proxy: does the retrieved text contain the response tokens (for faithful rows)?
        from common import toks
        if row["label"] == 1:
            rt = toks(row["response_bn"])
            cov = len(rt & toks(joined)) / len(rt) if rt else 0
            hit += cov
        print(f"Q: {row['prompt_bn'][:55]}")
        print(f"   top: [{res[0][0]:.2f}] {res[0][1]} :: {res[0][2][:90]}")
    print(f"\nmean response-coverage on faithful rows: {hit/sum(1 for r_ in S[:15] if r_['label']==1):.2f}")
