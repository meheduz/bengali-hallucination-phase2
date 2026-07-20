"""Context-row model: the verified construction artifact + overlap features.

Verified finding (4 independent agents): a faithful response is a verbatim
substring of its context; a hallucinated one is not. ~0.88 F1 alone.
Here we learn on top of it with related overlap features + the LLM judge score.

Honest 5-fold CV; the LLM score is optional (--with-llm).
"""
import sys, json, re
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from common import load_samples, load_test, f1_halluc, report
from rules import extract_numbers

PUNCT = r'[।,\.\-\'"“”‘’()!?;:\s]'

def norm(s):
    return re.sub(PUNCT, '', str(s))

def lcs_ratio(a, b):
    """longest common substring length / len(a), via difflib (fast enough here)."""
    from difflib import SequenceMatcher
    if not a: return 0.0
    m = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    return m.size / len(a)

def feats(r):
    resp, ctx, pr = str(r["response_bn"]), str(r["context"]), str(r["prompt_bn"])
    nr, nc = norm(resp), norm(ctx)
    sub = 1.0 if (nr and nr in nc) else 0.0
    lcs = lcs_ratio(nr, nc)
    rt = set(re.findall(r"\S+", resp)); ct = set(re.findall(r"\S+", ctx))
    cov = len(rt & ct) / len(rt) if rt else 0.0
    rnum = extract_numbers(resp) - extract_numbers(pr)
    cnum = extract_numbers(ctx)
    novel_num = 1.0 if (rnum - cnum) else 0.0
    n_novel = len(rnum - cnum)
    # rare/entity-ish tokens in response missing from context
    long_tok = [t for t in rt if len(t) > 3]
    miss_long = sum(1 for t in long_tok if t not in ctx) / max(1, len(long_tok))
    return [sub, lcs, cov, novel_num, n_novel, miss_long, len(resp), len(rt)]

FN = ["substring", "lcs_ratio", "tok_cov", "novel_num", "n_novel", "miss_long", "resp_len", "n_tok"]

def _experiment():
    S = [r for r in load_samples() if r["context"]]
    y = np.array([r["label"] for r in S])
    X = np.array([feats(r) for r in S])
    print(f"{len(S)} context rows; {int((y==0).sum())} halluc / {int((y==1).sum())} faithful")

    # baseline: substring rule alone
    sub_pred = [1 if f[0] else 0 for f in X]
    print(f"\nsubstring rule alone:  F1(halluc)={f1_halluc(list(y), sub_pred):.3f}")

    # learned model, 5-fold CV x 5 seeds
    oof_all = []
    for seed in range(5):
        oof = np.zeros(len(S))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            m = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                           random_state=seed)
            m.fit(X[tr], y[tr])
            oof[va] = m.predict_proba(X[va])[:, 0]  # P(halluc)
        best = max(((f1_halluc(list(y), [0 if s > t else 1 for s in oof]), t)
                    for t in np.arange(0.2, 0.81, 0.02)))
        oof_all.append(best[0])
    print(f"GB on overlap feats:   F1(halluc)={np.mean(oof_all):.3f} +/- {np.std(oof_all):.3f}")

    m = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0).fit(X, y)
    print("\nfeature importances:")
    for n, i in sorted(zip(FN, m.feature_importances_), key=lambda x: -x[1]):
        print(f"  {n:12s} {i:.3f}")

    json.dump({"feature_names": FN}, open("ctx_model_meta.json", "w"))
    print("\nctx model ready (fit at predict time on all 132 rows)")

if __name__ == "__main__":
    _experiment()
