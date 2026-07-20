"""Shared data loading + evaluation for the Bengali hallucination task.

Metric per rules: binary F1 on the HALLUCINATED class (label == 0).
We report both that and macro-F1 (Overview mentions macro-F1).
"""
import json, csv, re, os, statistics

DATA = os.path.join(os.path.dirname(__file__), "..", "bengali-hallucination")

def _clean(r):
    for k in ("prompt_bn", "response_bn", "context"):
        r[k] = "" if r.get(k) is None else str(r[k])
    if r["context"].strip() in ("[NULL]", ""):
        r["context"] = ""
    return r

def load_samples():
    d = json.load(open(os.path.join(DATA, "dataset samples.json")))
    return [_clean(dict(r)) for r in d]

def load_test():
    rows = list(csv.DictReader(open(os.path.join(DATA, "test set.csv"))))
    return [_clean(dict(r)) for r in rows]

BN = re.compile(r"[ঀ-৿]+")
NUM = re.compile(r"[০-৯0-9]+")

def toks(s):
    return set(BN.findall(s))

def f1_halluc(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0

def f1_macro(y_true, y_pred):
    def f1(pos):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == pos and p == pos)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != pos and p == pos)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == pos and p != pos)
        pr = tp/(tp+fp) if tp+fp else 0.0
        rc = tp/(tp+fn) if tp+fn else 0.0
        return 2*pr*rc/(pr+rc) if pr+rc else 0.0
    return (f1(0) + f1(1)) / 2

def report(y_true, y_pred, name=""):
    print(f"{name:28s} F1(halluc)={f1_halluc(y_true,y_pred):.3f}  macroF1={f1_macro(y_true,y_pred):.3f}  "
          f"acc={sum(a==b for a,b in zip(y_true,y_pred))/len(y_true):.3f}")
