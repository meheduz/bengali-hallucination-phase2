"""ctx_bio — Bengali date-format normalization for the ctx_gold matcher.

TARGET C ASSIGNMENT was: re-attack the 236 ctx_think rows @0.875, on the theory
that the 156 wiki-biography birth rows hide ~30 recoverable errors.

RESULT ON THE ASSIGNED TARGET: nothing to ship.  Three independent signals say
the judge is already correct on that pool (see analysis/ctx_bio.md).

WHAT THIS FILE DOES SHIP is a single incidental fix found on the way: the
ctx_gold tier scores a response HALLUCINATED when it differs from the gold
answer only by a Bengali ordinal suffix.

    Schematic of the failure (synthetic example, not competition data):

    Q     <person> কবে জন্মগ্রহণ করেন?
    CTX   <person> (<D>রা <month>, <year> - ...) ...
                    ^^^^^^^^^^^^^^^^^^^^ response is verbatim THIS span
    RESP  <D>রা <month>, <year>
    GOLD  <D> <month>, <year>       <- differs only by the ordinal "রা"
    ctx_gold pred_label = 0  ->  must be 1

    Fires on exactly one wiki-biography birth-date row in the test split.

The rule: a ctx_gold row scored 0 flips to 1 iff the response equals a gold
answer after date normalization (ordinal suffixes, era words, Bengali->ASCII
digits, punctuation).  Equality with the GOLD ANSWER -- deliberately not
containment in the context, which is far looser and produces false rescues.

Fires on exactly 1 of the 380 ctx_gold rows scored hallucinated.  Zero
parameters, no labels, no retrieval.
"""
import json, os, re, sys, unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_test

HERE = os.path.dirname(os.path.abspath(__file__))
BN2A = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# A Bengali ordinal suffix binds to a preceding numeral: ১লা ২রা ৪ঠা ৭ই ২৫শে.
#
# \b MUST NOT be used to close this match.  A Bengali ordinal ends in a vowel
# sign such as "া" (U+09BE), Unicode category Mn, which Python's re does not
# count as a word character -- so \b never fires after it and the suffix is
# silently left in place.  That bug is exactly why this row was missed before.
# Use an explicit follow-set instead.
ORD = re.compile(r"(?<=[০-৯0-9])\s*(তারিখ|য়ে|তম|লা|রা|ঠা|শে|ই)(?=\s|$|[,\.।;:\)\-–—])")
ERA = re.compile(r"খ্রিস্টাব্দ|খ্রিষ্টাব্দ|খ্রীষ্টাব্দ|বঙ্গাব্দ|সাল|সন|অব্দ|তারিখ")
MONTH_VAR = {"জানুয়ারী": "জানুয়ারি", "ফেব্রুয়ারী": "ফেব্রুয়ারি", "আগষ্ট": "আগস্ট",
             "সেপ্টেম্বার": "সেপ্টেম্বর", "অক্টোবার": "অক্টোবর",
             "নভেম্বার": "নভেম্বর", "ডিসেম্বার": "ডিসেম্বর"}
DIGIT = re.compile(r"[০-৯0-9]")


def norm_date(s):
    s = unicodedata.normalize("NFC", str(s))
    s = ORD.sub("", s)
    for a, b in MONTH_VAR.items():
        s = s.replace(a, b)
    s = ERA.sub(" ", s)
    s = re.sub(r"[েিোাৈৌীূুৃ](?=\s|$)", " ", s)   # dangling vowel left by ERA
    s = s.translate(BN2A)
    s = re.sub(r"(?<!\d)0+(\d)", r"\1", s)
    s = re.sub(r"[,\.।;:\-–—/‘’'\"()\[\]]", " ", s)
    return re.sub(r"\s+", "", s)


def rescue(i, m, row):
    """1 if this ctx_gold hallucinated row is a date-format false negative."""
    if m.get("pred_label") != 0 or m.get("suspect_gold"):
        return None
    resp = row["response_bn"]
    if not DIGIT.search(resp):
        return None
    golds = m.get("gold_answers") or ([m["gold_answer"]] if m.get("gold_answer") else [])
    nr = norm_date(resp)
    if len(nr) < 5:
        return None
    for g in golds:
        ng = norm_date(g)
        if ng and nr == ng:
            return 1
    return None


OUT = "source_match_ctx_bio.json"


def main():
    T = load_test()
    cm = json.load(open(os.path.join(HERE, "source_match_ctx.json")))
    out = []
    for k, m in cm.items():
        i = int(k)
        if rescue(i, m, T[i]) == 1:
            out.append({"i": i, "pred": 1, "tier": "ctx_bio_dateformat",
                        "was": m["pred_label"], "resp": T[i]["response_bn"],
                        "gold": m.get("gold_answer")})
    n0 = sum(1 for m in cm.values() if m.get("pred_label") == 0 and not m.get("suspect_gold"))
    print(f"ctx_gold rows scored hallucinated: {n0}")
    print(f"date-format false negatives rescued: {len(out)}")
    for o in out:
        print(f"  idx={o['i']}  0 -> 1   resp={o['resp']!r}  gold={o['gold']!r}")
    json.dump(out, open(os.path.join(HERE, OUT), "w"), ensure_ascii=False, indent=1)
    print(f"wrote {OUT}")
    print("PRECEDENCE: must be applied ABOVE ctx_gold in the builder.")


if __name__ == "__main__":
    main()
