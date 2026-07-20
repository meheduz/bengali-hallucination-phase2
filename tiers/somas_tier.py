"""Deterministic সমাস tier for the authored-ctx grammar block.

The authored context SUPPLIES the ব্যাসবাক্য (schematically:
"'<compound>' শব্দটির ব্যাসবাক্য হলো: <expansion>।") and the question asks
which সমাস it is. Standard NCTB
Bengali grammar makes that mapping DETERMINISTIC from the ব্যাসবাক্য's case
marking / joining particle -- no world knowledge, no labels, no retrieval.

This matters because ctx_think (a passage-faithfulness judge) answers
"faithful" on 48 of these 49 rows: the response never contradicts the context,
because the context does not state the answer at all. The judge is
structurally mismatched to the row type.
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_test

NUMW = r"(?:এক|দুই|দু|তিন|চার|পাঁচ|ছয়|সাত|আট|নয়|দশ|শত|সহস্র|নব|ত্রি|চতুর|পঞ্চ|ষট|সপ্ত|অষ্ট)"


def byas_of(ctx):
    m = re.search(r"ব্যাসবাক্য হলো:?\s*(.+?)\s*।?\s*$", ctx.strip())
    return m.group(1).strip() if m else None


def somas_of(b):
    """ব্যাসবাক্য -> সমাসের নাম. Returns None where the canon is ambiguous."""
    s = re.sub(r"\s*\([^)]*\)\s*", " ", b).strip()   # drop glosses like (রাবণ)

    # --- বহুব্রীহি family (possessive: "... যার/যাহার") ---------------------
    if re.search(r"\bপরস্পর\b", s) and re.search(r"দ্বারা|দ্বার", s):
        return "ব্যতিহার বহুব্রীহি"
    if re.search(r"(?:^|\s)(?:নাই|নেই)\s", s) and re.search(r"যার|যাহার", s):
        return "নঞ বহুব্রীহি"
    if re.search(r"(?:যার|যাহার|যাঁর)\s*$", s):
        return "বহুব্রীহি"

    # --- দ্বিগু (numeral + সমাহার) ------------------------------------------
    if re.search(NUMW + r".*সমাহার", s):
        return "দ্বিগু"

    # --- নঞ তৎপুরুষ ---------------------------------------------------------
    if re.match(r"^(?:না|নয়|অ|অন)\s", s):
        return "নঞ তৎপুরুষ"

    # --- দ্বন্দ্ব (X ও Y) ---------------------------------------------------
    if re.search(r"\sও\s", s):
        return "দ্বন্দ্ব"

    # --- কর্মধারয় family ----------------------------------------------------
    if re.search(r"-?ই\s", s):
        return "রূপক কর্মধারয়"
    if re.search(r"ন্যায়|মতো|মত\b", s):
        # "X-এর ন্যায় <quality>"  -> উপমান ; "<thing> X-এর ন্যায়" -> উপমিত
        if re.search(r"ন্যায়\s*$", s):
            return "উপমিত কর্মধারয়"
        return "উপমান কর্মধারয়"
    if re.search(r"চিহ্নিত|বিশিষ্ট", s):
        return "মধ্যপদলোপী কর্মধারয়"
    if re.search(r"\sযে\s", s):
        return "কর্মধারয়"

    # --- অব্যয়ীভাব ----------------------------------------------------------
    if re.search(r"অনুযায়ী|ব্যাপিয়া|ধরিয়া|পর্যন্ত\s*ব্যাপ্ত|যথা|সদৃশ|অভাব", s):
        return "অব্যয়ীভাব"

    # --- তৎপুরুষ by case marker ---------------------------------------------
    if re.search(r"হইতে|হতে|থেকে", s):
        return "পঞ্চমী তৎপুরুষ"
    if re.search(r"জন্য|নিমিত্ত", s):
        return "চতুর্থী তৎপুরুষ"
    if re.search(r"দ্বারা|দিয়া|দিয়ে|কর্তৃক", s):
        return "তৃতীয়া তৎপুরুষ"
    if re.search(r"কে\s", s):
        return "দ্বিতীয়া তৎপুরুষ"
    if re.search(r"(?:ের|র)\s", s):
        return "ষষ্ঠী তৎপুরুষ"
    if re.search(r"(?:ে|তে|য়)\s", s):
        return "সপ্তমী তৎপুরুষ"
    return None


HEADS = ["ব্যতিহার বহুব্রীহি", "নঞ বহুব্রীহি", "বহুব্রীহি", "দ্বিগু",
         "নঞ তৎপুরুষ", "দ্বন্দ্ব", "রূপক কর্মধারয়", "উপমান কর্মধারয়",
         "উপমিত কর্মধারয়", "মধ্যপদলোপী কর্মধারয়", "কর্মধারয়", "অব্যয়ীভাব",
         "পঞ্চমী তৎপুরুষ", "চতুর্থী তৎপুরুষ", "তৃতীয়া তৎপুরুষ",
         "দ্বিতীয়া তৎপুরুষ", "ষষ্ঠী তৎপুরুষ", "সপ্তমী তৎপুরুষ", "তৎপুরুষ"]


def resp_somas(resp):
    r = re.sub(r"\s+", " ", str(resp)).strip()
    for h in HEADS:              # longest/most specific first
        if h in r:
            return h
    return None


def judge(ctx, resp):
    b = byas_of(ctx)
    if not b:
        return None, "no_byas", None
    truth = somas_of(b)
    if truth is None:
        return None, "ambiguous_byas", b
    said = resp_somas(resp)
    if said is None:
        return None, "no_somas_in_resp", b
    if said == truth:
        return 1, f"exact:{truth}", b
    # a bare family name when the canon demands a subtype is NOT the same answer
    return 0, f"want:{truth}|said:{said}", b


if __name__ == "__main__":
    T = load_test()
    sm = json.load(open(os.path.join(os.path.dirname(__file__), "source_match_ctx.json")))
    good = {int(k) for k, v in sm.items()
            if v.get("pred_label") in (0, 1) and not v.get("suspect_gold")}
    rows = [i for i, r in enumerate(T)
            if r["context"] and i not in good and "ব্যাসবাক্য" in r["context"]]
    tk = json.load(open(os.path.join(os.path.dirname(__file__), "ctx_think_test.json")))
    ct = dict(zip(tk["idx"], tk["pred"]))
    out, agree, n = [], 0, 0
    for i in rows:
        p, how, b = judge(T[i]["context"], T[i]["response_bn"])
        if p is None:
            print(f"ABSTAIN {i} {how} :: {b}")
            continue
        n += 1
        agree += (p == ct.get(i))
        out.append({"i": i, "pred": p, "byas": b, "how": how})
        print(f"{i} pred={p} think={ct.get(i)} {'SAME' if p==ct.get(i) else 'FLIP'} :: {b} -> {how}")
    print(f"\ndecided {n}/{len(rows)}; agrees with ctx_think on {agree} "
          f"=> {n-agree} FLIPS")
    print("pred dist:", {v: sum(1 for o in out if o['pred'] == v) for v in (0, 1)})
    json.dump(out, open(os.path.join(os.path.dirname(__file__),
              "source_match_ctx_somas.json"), "w"), ensure_ascii=False, indent=1)


# --------------------------------------------------------------- সন্ধি tier
SANDHI_CTX = re.compile(r"'?([^'\s]+)'?\s*ও\s*'?([^'\s]+)'?\s*শব্দ দুটির")


def sandhi_judge(ctx, resp):
    """সন্ধি joins two words with a REQUIRED phonological change at the
    junction. If the response is the two words concatenated verbatim, no
    sandhi was performed -> hallucinated. Only this unambiguous direction is
    decided; anything else abstains."""
    m = SANDHI_CTX.search(ctx)
    if not m:
        return None, "no_parse", None
    a, b = m.group(1).strip("'‘’"), m.group(2).strip("'‘’")
    # বিসর্গ সন্ধি is নিপাতনে-irregular: দুঃ+খ -> দুঃখ legitimately IS the plain
    # concatenation (visarga retained), while নিঃ+কাম -> নিষ্কাম is not.
    # Excluding visarga-final first words keeps the detector one-directional.
    if a.endswith("ঃ"):
        return None, "visarga_irregular", f"{a}+{b}"
    r = re.sub(r"[\s'‘’।]", "", str(resp))
    if r == a + b:
        return 0, f"no_sandhi_applied:{a}+{b}", f"{a}+{b}"
    return None, "changed_junction", f"{a}+{b}"
