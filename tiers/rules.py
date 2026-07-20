"""Deterministic rule signals from error analysis.

- numeric_veto: for ctx rows, response numbers must appear in the context
  (Bengali/ASCII digits normalized, common Bengali number-words mapped).
  Returns True -> hard evidence of hallucination.
- is_math_prompt: closed-book arithmetic word problems, where the LLM judge's
  yes/no verdict is noise -> route to thinking-mode generation.
"""
import re

BN2ASCII = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

NUM_WORDS = {
    "এক": 1, "দুই": 2, "তিন": 3, "চার": 4, "পাঁচ": 5, "ছয়": 6, "সাত": 7,
    "আট": 8, "নয়": 9, "দশ": 10, "এগারো": 11, "বারো": 12, "তেরো": 13,
    "চৌদ্দ": 14, "পনেরো": 15, "ষোল": 16, "সতেরো": 17, "আঠারো": 18,
    "উনিশ": 19, "বিশ": 20, "ত্রিশ": 30, "চল্লিশ": 40, "পঞ্চাশ": 50,
    "ষাট": 60, "সত্তর": 70, "আশি": 80, "নব্বই": 90, "শত": 100, "একশ": 100,
    "হাজার": 1000, "লক্ষ": 100000, "লাখ": 100000, "কোটি": 10000000,
}

def extract_numbers(text):
    t = text.translate(BN2ASCII)
    nums = set(int(x) for x in re.findall(r"\d+", t))
    for w, v in NUM_WORDS.items():
        if w in text:
            nums.add(v)
    return nums

def numeric_veto(response, context):
    """True if the response contains a number that the context does not."""
    if not context:
        return False
    rn = extract_numbers(response)
    if not rn:
        return False
    cn = extract_numbers(context)
    return bool(rn - cn)

MATH_PAT = re.compile(
    r"(কত টাকা|শতকরা|লাভ|ক্ষতি|সুদ|আসল|গড়|অনুপাত|যোগফল|বিয়োগফল|গুণফল|ভাগফল"
    r"|কত অংশ|মোট কত|বয়সের|বয়স কত|গতিবেগ|কত সময়|কত দিন|কত ঘন্টা|সমষ্টি"
    r"|বিক্রয়মূল্য|ক্রয়মূল্য|আয়তন কত|ক্ষেত্রফল|পরিসীমা|শতাংশ)")

def is_math_prompt(prompt, context=""):
    """Closed-book arithmetic word problem detector."""
    if context:
        return False
    p = prompt
    has_kw = bool(MATH_PAT.search(p))
    n_nums = len(re.findall(r"[\d০-৯]+", p))
    return has_kw and n_nums >= 1 or n_nums >= 3

if __name__ == "__main__":
    from common import load_samples, f1_halluc
    S = load_samples()
    # numeric veto precision on ctx rows
    ctx = [r for r in S if r["context"]]
    tp = fp = 0
    for r in ctx:
        if numeric_veto(r["response_bn"], r["context"]):
            if r["label"] == 0: tp += 1
            else: fp += 1
    print(f"numeric_veto on ctx rows: fires {tp+fp}, correct(halluc) {tp}, wrong(faithful) {fp}")
    # math detector coverage
    nb = [r for r in S if not r["context"]]
    m = [r for r in nb if is_math_prompt(r["prompt_bn"])]
    print(f"math detector: {len(m)}/{len(nb)} closed-book rows flagged; labels: "
          f"{sum(1 for r in m if r['label']==0)} halluc / {sum(1 for r in m if r['label']==1)} faithful")
