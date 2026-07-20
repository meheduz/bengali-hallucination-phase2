"""Rule-based Bengali-grammar tier for closed-book rows.

Uses work/assets/bn_grammar_kb.json (NCTB canon). Decides only the categories
where the canon is unambiguous: শুদ্ধ বানান MCQ, বিপরীতার্থক শব্দ, সমাস নাম।
Everything else -> abstain (None).
"""
import json, re, os, csv, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
KB = json.load(open(os.path.join(ROOT, "assets", "bn_grammar_kb.json")))

SPELL_OK = {k for k in KB["spelling"] if not k.startswith("_")}
SPELL_BAD = {}
for good, bads in KB["spelling"].items():
    if good.startswith("_"):
        continue
    for b in bads:
        SPELL_BAD[b] = good
OPP = {k: v for k, v in KB["opposites"].items() if not k.startswith("_")}
SOMAS = {k: v for k, v in KB["somas"].items() if not k.startswith("_") and k != "notes"}

OPT = re.compile(r"[কখগঘ]\)\s*([^,;\n।]+)")
PUNCT = "।,;:?!'\"‘’“”()-— \t\n।"


def norm(s):
    return (s or "").strip().strip(PUNCT).strip()


def resolve_option(prompt, resp):
    """If the response is just an option letter, map to the option text."""
    r = norm(resp)
    m = re.fullmatch(r"([কখগঘ])\)?", r)
    if not m:
        return r
    opts = OPT.findall(prompt)
    letters = re.findall(r"([কখগঘ])\)", prompt)
    d = dict(zip(letters, [norm(o) for o in opts]))
    return d.get(m.group(1), r)


def match(prompt, resp):
    p, r0 = str(prompt or ""), str(resp if resp is not None else "")
    ans = resolve_option(p, r0)
    toks = re.findall(r"[ঀ-৿]+", ans)

    # 1) শুদ্ধ বানান
    if re.search(r"(শুদ্ধ\s*বানান|বানান\s*(টি|টা)?\s*শুদ্ধ|বানানটি\s*শুদ্ধ)", p) and "অশুদ্ধ" not in p:
        for t in toks:
            if t in SPELL_OK:
                return 1, "spell_ok:" + t
            if t in SPELL_BAD:
                return 0, "spell_bad:%s->%s" % (t, SPELL_BAD[t])
        return None, ""

    # 2) বিপরীতার্থক শব্দ
    if re.search(r"বিপরীত(ার্থক)?", p):
        for w, goods in OPP.items():
            if w in p:
                if any(g in ans for g in goods):
                    return 1, "opp_ok:%s=%s" % (w, goods[0])
                if toks:
                    return 0, "opp_bad:%s!=%s (got %s)" % (w, goods[0], ans)
        return None, ""

    # 3) সমাস: "কোনটি X সমাসের উদাহরণ"
    m = re.search(r"কোনটি\s*[‘'\"]?(\S+?)[’'\"]?\s*সমাস", p)
    if m and "উদাহরণ" in p:
        want = m.group(1).strip()
        for t in toks:
            if t in SOMAS:
                got = SOMAS[t]
                ok = want in got
                return (1 if ok else 0), "somas:%s=%s want=%s" % (t, got, want)
        return None, ""

    return None, ""


if __name__ == "__main__":
    S = json.load(open(os.path.join(ROOT, "..", "bengali-hallucination", "dataset samples.json")))
    cb = [s for s in S if (s["context"] or "").strip() in ("", "[NULL]")]
    n = ok = 0
    for s in cb:
        pred, how = match(s["prompt_bn"], s["response_bn"])
        if pred is None:
            continue
        n += 1
        good = pred == int(s["label"])
        ok += good
        print(("OK " if good else "BAD"), pred, s["label"], how, "|", s["prompt_bn"][:60], "|", s["response_bn"][:40])
    print("validated n=%d acc=%.3f" % (n, ok / n if n else 0))
