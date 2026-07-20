"""Deterministic বাগধারা/ভাবার্থ + উপসর্গ tiers for the authored-ctx block.

Both families share the সমাস precedent's shape: the authored *context* does not
state the answer, so a passage-faithfulness judge has nothing to check and
defaults to "faithful".  The answer is instead fixed by an OFFLINE CANON that
predates the test set --- for উপসর্গ a closed set of morphemes, for বাগধারা a
dictionary gloss.

  উপসর্গ.  Bengali উপসর্গ form three CLOSED classes (সংস্কৃত ২০, খাঁটি বাংলা ২১,
  বিদেশি).  The context names the উপসর্গ verbatim; the question asks its class.
  If the named উপসর্গ is absent from the class the response asserts, the
  response is wrong by enumeration -- no world knowledge involved.

  বাগধারা.  The response gives a meaning for a named idiom.  We resolve the
  idiom's gloss from three independent offline sources (bengali_idioms.json,
  bn.wiktionary dump, mined exam Q/A) and decide by HEAD-CONCEPT AGREEMENT with
  a POLARITY GUARD:

      pred = 1  iff  response and gloss share >=1 content head
                     AND no polarity axis puts them on opposite sides
      pred = 0  iff  they share no content head, OR a polarity axis conflicts
      abstain   otherwise (no gloss / self-reuse only / gloss-by-synonym-idiom)

  The polarity guard exists because this family's hallucinations are built by
  INVERTING the gloss, which leaves lexical overlap intact:

      দুধের মাছি   resp "দুঃসময়ের বন্ধু"      gloss "সুসময়ের বন্ধু"   overlap 1.00
      উলুবনে ...   resp "উপযুক্ত পাত্রে দান"  gloss "অপাত্রে দান"     overlap 0.67

  Overlap alone would call both faithful.  The axes below are ordinary Bengali
  oppositions (সামান্য/সম্পূর্ণ, সু-/দুঃ-, সাফল্য/সর্বনাশ, ...); they are not
  fitted to any label, but note in the writeup that WHICH axes are included was
  chosen after reading these rows.
"""
import json, os, re, sys, glob

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load_test

BN = re.compile(r"[ঀ-৿]+")


def norm(s):
    return re.sub(r"[^ঀ-৿]", "", str(s))


# ===================================================================== উপসর্গ
KB = json.load(open(os.path.join(HERE, "assets", "bn_grammar_kb.json")))

# The KB's বাংলা list is defective (it claims ২১ but repeats 'হা' and omits হর,
# আম, ঊন), and its বিদেশি list omits নিম.  Union it with the standard NCTB
# enumeration so the "closed set" argument is actually sound -- a membership
# test is only valid if the sets are complete.
NCTB = {
    "সংস্কৃত": "প্র পরা অপ সম নি অনু অব নির্ নির দুর্ দুর বি অধি সু উৎ পরি প্রতি অতি অপি অভি উপ আ",
    "বাংলা":   "অ অনা অজ আ আন আব ইতি ঊন উন কদ কু নি পাতি বি ভর রাম স সা সু হা আম হর আড়",
    "বিদেশি":  "কার কম বদ বে বর দর নিম ফি লা গর হর হাফ না গের সাব হেড ফুল খাস বাজে",
}
UP = {k: set(KB["upasarga"].get(k, [])) | set(v.split()) for k, v in NCTB.items()}

# how a response names each class
CLASS_WORDS = {
    "সংস্কৃত": ["সংস্কৃত", "তৎসম"],
    "বাংলা":   ["খাঁটি বাংলা", "বাংলা"],
    "বিদেশি":  ["বিদেশি", "বিদেশী", "ফারসি", "ফার্সি", "আরবি", "ইংরেজি", "উর্দু", "হিন্দি"],
}

UP_CTX = re.compile(r"'?([^'\s]+)'?\s*উপসর্গ")
UP_Q = re.compile(r"'([^']+)'\s*উপসর্গ")


def upasarga_of(ctx, q):
    m = UP_Q.search(q) or UP_CTX.search(ctx)
    return m.group(1).strip("'‘’\"") if m else None


def resp_class(resp):
    """Which class does the response name?  'বাংলা' is checked last because
    'খাঁটি বাংলা' contains it and বিদেশি answers often gloss a source language."""
    r = str(resp)
    for cls in ("বিদেশি", "সংস্কৃত"):
        if any(w in r for w in CLASS_WORDS[cls]):
            return cls
    if "বাংলা" in r:
        return "বাংলা"
    return None


def judge_upasarga(ctx, q, resp):
    u = upasarga_of(ctx, q)
    if not u:
        return None, "no_upasarga_in_ctx", None
    said = resp_class(resp)
    if said is None:
        return None, "no_class_in_resp", u
    member = [c for c in ("সংস্কৃত", "বাংলা", "বিদেশি") if u in UP[c]]
    if not member:
        # not in ANY closed list -> the asserted class is wrong by enumeration
        # (only sound because all three lists are closed and complete)
        return 0, f"{u}:not_in_any_closed_set|said:{said}", u
    if said in member:
        tag = "unique" if len(member) == 1 else "multi:" + "/".join(member)
        return 1, f"{u}:in_{said}({tag})", u
    return 0, f"{u}:in_{'/'.join(member)}|said:{said}", u


# ==================================================================== বাগধারা
CANON = json.load(open(os.path.join(HERE, "assets", "bengali_idioms.json")))["idioms"]
NC = {norm(k): (k, v) for k, v in CANON.items()}
WK = json.load(open(os.path.join(HERE, "wikt_pages.json")))
NW = {norm(k): (k, v) for k, v in WK.items()}


def _wikt_gloss(page):
    txt = str(page)
    m = (re.search(r"===\s*ভাবার্থ\s*===(.*?)(?:===|\Z)", txt, re.S)
         or re.search(r"==\s*অর্থ\s*==(.*?)(?:==[^=]|\Z)", txt, re.S)
         or re.search(r"===\s*(?:বিশেষণ|ক্রিয়া|বিশেষ্য)\s*===(.*?)(?:===|\Z)", txt, re.S))
    if not m:
        return None
    lines = []
    for ln in m.group(1).split("\n"):
        ln = ln.strip()
        if not ln.startswith(("#", "*")) or ln.startswith(("#:", "#*")):
            continue
        ln = re.sub(r"\{\{[^}]*\}\}", " ", ln)
        ln = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", ln)
        ln = ln.lstrip("#* ").strip(" ।")
        if ln and BN.search(ln):
            lines.append(ln)
    return ", ".join(lines[:3]) or None


_EXAM = None


def exam_pairs():
    global _EXAM
    if _EXAM is None:
        E = []
        for r in json.load(open(os.path.join(HERE, "source_hunt", "livemcq", "qa2.json"))):
            E.append((r.get("q", ""), r.get("ans", "")))
        for p in glob.glob(os.path.join(HERE, "source_hunt", "sites", "*", "*.json")):
            try:
                d = json.load(open(p))
            except Exception:
                continue
            if isinstance(d, list):
                for r in d:
                    if isinstance(r, dict):
                        E.append((str(r.get("q", "")), str(r.get("ans", ""))))
        _EXAM = [(norm(q), a, q) for q, a in E]
    return _EXAM


def glosses(phrase):
    """Return [(source, key, gloss)].  Exact-key hits are trusted alone; a
    sub-phrase (fuzzy) hit is admitted ONLY if a second independent source
    agrees with it, because dropping a word can change the idiom outright
    ('লাল বাতি জ্বলা' vs the wiktionary entry 'লালবাতি' = অনুমতি নেই)."""
    nk = norm(phrase)
    exact, fuzzy = [], []
    if nk in NC:
        exact.append(("canon", NC[nk][0], NC[nk][1]))
    if nk in NW:
        g = _wikt_gloss(NW[nk][1])
        if g:
            exact.append(("wikt", NW[nk][0], g))
    for a in [re.sub(r"^[কখগঘ]\)\s*", "", a).strip()
              for nq, a, q in exam_pairs()
              if nk and nk in nq and ("অর্থ" in q or "বাগধারা" in q or "প্রবাদ" in q) and a.strip()][:3]:
        exact.append(("exam", phrase, a))
    if exact:
        return exact
    if len(nk) > 6:
        for src, tbl, get in (("canon", NC, lambda v: v[1]), ("wikt", NW, lambda v: _wikt_gloss(v[1]))):
            for k2, v in tbl.items():
                if k2 != nk and len(k2) > 5 and k2 in nk:
                    g = get(v)
                    if g:
                        fuzzy.append((src + "~", v[0], g))
    if len(fuzzy) >= 2:
        # require the fuzzy candidates to corroborate each other
        best = max(fuzzy, key=lambda f: sum(agree(f[2], o[2]) for o in fuzzy if o is not f))
        if any(agree(best[2], o[2]) >= 0.34 or not polarity_conflict(best[2], o[2])[0]
               for o in fuzzy if o is not best and o[0] != best[0]):
            return [f for f in fuzzy if f[0] != best[0] or f is best][:2]
    return []


# ---- token machinery -------------------------------------------------------
STOP = set("""ও বা এবং করা হওয়া কোনো কোন এমন যে যা তার তাই আর নিয়ে থেকে সঙ্গে সাথে
ব্যক্তি লোক লোকের বস্তু জিনিস অবস্থা কিছু ভাব মানুষ ধরনের হয় হয়ে করে যিনি যার
না নেই নয় থাকা পারা""".split())
# pure intensity/degree modifiers: they carry no head concept, so an overlap on
# them alone is not agreement ("হঠাৎ ধনী" vs "হঠাৎ বিস্মিত")
MOD = set("""হঠাৎ নিতান্ত অত্যন্ত অতি খুব খুবই ভীষণ সম্পূর্ণ সবচেয়ে বেশি বেশ বড় চরম
অতিশয় একেবারে খুবি প্রায় আরও সব দুই দুদিক""".split())
SUF = ["গুলোর", "গুলির", "গুলো", "গুলি", "দেরকে", "দের", "টির", "টাকে", "কেই",
       "ের", "রা", "কে", "তে", "টি", "টা", "য়", "র", "ে"]


def ortho(t):
    """Light orthographic normalization: বাংলা spelling varies freely between
    ী/ি and ূ/ু (বেনামী ~ বেনামি), which would otherwise block a match."""
    return t.replace("ী", "ি").replace("ূ", "ু")


def stem(t):
    t = ortho(t)
    for s in SUF:
        if len(t) > len(s) + 2 and t.endswith(ortho(s)):
            return t[: -len(s)]
    return t


def tokens(s):
    return {ortho(t) for t in BN.findall(str(s))}


def heads(s):
    return {stem(t) for t in BN.findall(str(s))
            if t not in STOP and stem(t) not in MOD and t not in MOD and stem(t) not in STOP}


def _same(x, y):
    return x == y or (len(x) >= 4 and x in y) or (len(y) >= 4 and y in x)


def shared_heads(a, b):
    A, B = heads(a), heads(b)
    return {x for x in A if any(_same(x, y) for y in B)}


def agree(a, b):
    A, B = heads(a), heads(b)
    if not A or not B:
        return 0.0
    return len(shared_heads(a, b)) / min(len(A), len(B))


# ---- polarity axes ---------------------------------------------------------
# Ordinary Bengali oppositions.  A conflict fires when the response sits
# exclusively on one side and the gloss exclusively on the other.
AXES = [
    ("degree",   ["সামান্য", "তুচ্ছ", "অল্প", "সীমিত", "নগণ্য", "অগভীর", "পাতলা"],
                 ["প্রকৃত", "প্রচুর", "বিপুল", "গভীর", "অগাধ", "সম্পূর্ণরূপে", "মর্মে", "ভরপুর"]),
    ("fortune",  ["সৌভাগ্য", "সৌভাগ্যবান", "ভাগ্যবান", "সুভাগ্য", "কপালজোর"],
                 ["হতভাগ্য", "মন্দভাগ্য", "দুর্ভাগ্য", "দুর্ভাগা", "অলক্ষুণে", "হতভাগা"]),
    ("outcome",  ["সাফল্য", "সফল", "সার্থক", "উন্নতি", "জয়", "লাভ"],
                 ["ব্যর্থ", "ব্যর্থতা", "সর্বনাশ", "বিপর্যয়", "ধ্বংস", "ভরাডুবি", "পরাজয়"]),
    # NB: 'সুন্দর' is deliberately NOT on the positive side -- contrast glosses
    # ("বাইরে সুন্দর ভিতরে অসার") would then hit both sides and cancel.
    ("worth",    ["উপযুক্ত", "যোগ্য", "সুপাত্র", "গুণসম্পন্ন", "গুণী"],
                 ["অপাত্র", "অযোগ্য", "অসার", "অন্তঃসারশূন্য", "অকেজো", "অপদার্থ", "নিষ্ফল"]),
    ("industry", ["কর্মঠ", "পরিশ্রমী", "কর্মী", "দক্ষ", "উদ্যমী"],
                 ["অলস", "কুঁড়ে", "অকর্মণ্য", "অকেজো", "অপদার্থ"]),
    ("thrift",   ["সঞ্চয়ী", "হিসাবি", "মিতব্যয়ী", "কৃপণ", "সঞ্চিত", "আগলে", "দুষ্প্রাপ্য"],
                 ["অমিতব্যয়ী", "বেহিসাবি", "অপচয়", "উদার", "সহজলভ্য", "সুলভ", "ব্যবহৃত"]),
    ("speed",    ["দ্রুত", "দ্রুতগতি", "ক্ষিপ্র", "তড়িৎ"],
                 ["ধীর", "মন্থর", "শ্লথ", "ঢিমে"]),
    ("time",     ["সুসময়", "সুদিন", "সাময়িক", "ক্ষণস্থায়ী"],
                 ["দুঃসময়", "দুর্দিন", "চির", "চিরস্থায়ী", "অনন্ত", "স্থায়ী"]),
    ("peril",    ["মুক্তি", "বিপদমুক্তি", "নিরাপদ", "উদ্ধার", "রক্ষা"],
                 ["আসন্ন", "সমূহ", "ঘনিয়ে", "সংক্রান্তি"]),
    ("order",    ["সুশাসিত", "সুশৃঙ্খল", "শান্ত", "নিয়মতান্ত্রিক"],
                 ["অরাজক", "অরাজকতা", "বিশৃঙ্খল", "যথেচ্ছাচার", "যথেচ্চারের"]),
    ("guile",    ["সরল", "সহজ", "নিষ্কপট", "সাধু"],
                 ["কুটিল", "কুটিলতা", "কূটবুদ্ধি", "কুচক্রী", "ধূর্ত", "প্যাঁচ"]),
    ("bond",     ["বন্ধুত্ব", "মিত্রতা", "সদ্ভাব", "প্রীতি"],
                 ["শত্রুতা", "বিবাদ", "সাপে", "নেউলে", "চিরশত্রু"]),
    ("reality",  ["বাস্তবসম্মত", "বাস্তব", "সম্ভব", "যুক্তিসঙ্গত"],
                 ["অসম্ভব", "অলীক", "কল্পনা", "আকাশকুসুম", "অবাস্তব"]),
    ("position", ["ভূমিকা", "অবতরণিকা", "সূচনা", "প্রারম্ভ", "শুরু"],
                 ["উপসংহার", "সমাপ্তি", "শেষাংশ", "পরিণতি"]),
    ("vitality", ["সুস্থ", "বেঁচে", "জীবিত", "আরোগ্য"],
                 ["মৃত্যু", "মারা", "মৃত", "প্রয়াণ"]),
]


def _side(text, words):
    """Match axis words as TOKENS, never as substrings.  Substring matching
    silently cancels every conflict this guard exists to catch: 'বেহিসাবি'
    contains 'হিসাবি', 'অগভীর' contains 'গভীর', 'পরিকল্পনা' contains
    'কল্পনা' -- so both sides light up and the conflict is suppressed."""
    tk = tokens(text) | heads(text)
    return {w for w in words if ortho(w) in tk or stem(w) in tk}


def polarity_conflict(resp, gloss):
    for name, A, B in AXES:
        ra, rb = _side(resp, A), _side(resp, B)
        ga, gb = _side(gloss, A), _side(gloss, B)
        if ra and gb and not rb and not ga:
            return True, f"{name}:{'/'.join(sorted(ra))}<>{'/'.join(sorted(gb))}"
        if rb and ga and not ra and not gb:
            return True, f"{name}:{'/'.join(sorted(rb))}<>{'/'.join(sorted(ga))}"
    return False, None


IDIOM_Q = re.compile(r"'([^']+)'")


def judge_idiom(q, resp):
    m = IDIOM_Q.search(q)
    if not m:
        return None, "no_phrase", None, None
    phrase = m.group(1).strip()
    G = glosses(phrase)
    if not G:
        return None, "no_gloss_in_canon", phrase, None
    # response that is itself a canon idiom headword is a gloss-by-synonymous-
    # idiom; lexical agreement cannot adjudicate it.
    if norm(resp) in NC and norm(resp) != norm(phrase):
        return None, "resp_is_canon_idiom", phrase, G[0]
    best = max(G, key=lambda g: agree(resp, g[2]))
    # A conflict against ANY sourced gloss is decisive -- the sources are
    # paraphrases of one meaning, and 'best' is picked by overlap, which is
    # exactly the statistic a polarity flip corrupts.
    for g in G:
        conf, why = polarity_conflict(resp, g[2])
        if conf:
            return 0, f"polarity[{why}]", phrase, g
    # a "shared head" that is just a word of the idiom itself is self-reuse,
    # not agreement with the gloss
    own = tokens(phrase) | heads(phrase)
    sh = set()
    for g in G:
        sh |= shared_heads(resp, g[2])
    sh_real = {x for x in sh if not any(_same(x, o) for o in own)}
    if sh_real:
        return 1, f"head_share:{'/'.join(sorted(sh_real))}", phrase, best
    # Response explains the idiom by restating its own words rather than by an
    # independent gloss ('তালকানা' -> "ছন্দ বা তাল বুঝতে না পারা").  Lexical
    # agreement cannot adjudicate that; abstain instead of guessing.
    # (>=3 chars on both sides: 'তা' is a substring of 'হতাশ' but not a reuse)
    own_big = {o for o in own if len(o) >= 3}
    if any(any(len(x) >= 3 and (x in o or o in x) for o in own_big) for x in heads(resp)):
        return None, "self_reuse_only", phrase, best
    if sh:
        return None, f"self_reuse_only:{'/'.join(sorted(sh))}", phrase, best
    return 0, "no_head_share", phrase, best


# ======================================================================= main
def rows_in_scope(T):
    sm = json.load(open(os.path.join(HERE, "source_match_ctx.json")))
    good = {int(k) for k, v in sm.items()
            if v.get("pred_label") in (0, 1) and not v.get("suspect_gold")}
    gi = {r["i"] for r in json.load(open(os.path.join(HERE, "source_match_ctx_grammar.json")))}
    return [i for i, r in enumerate(T) if r["context"] and i not in good and i not in gi]


if __name__ == "__main__":
    T = load_test()
    pool = rows_in_scope(T)
    try:
        tk = json.load(open(os.path.join(HERE, "ctx_think_test.json")))
        CT = dict(zip(tk["idx"], tk["pred"]))
    except Exception:
        CT = {}

    out, stats = [], {"idiom": [0, 0, 0], "upasarga": [0, 0, 0]}  # ship, abstain, flips
    for i in pool:
        q, c, r = T[i]["prompt_bn"], T[i]["context"], T[i]["response_bn"]
        if "উপসর্গ" in q or "উপসর্গ" in c:
            tier = "upasarga"
            p, how, key = judge_upasarga(c, q, r)
            ev = None
        elif "বাগধারা" in q or "প্রবাদ" in q:
            tier = "idiom"
            p, how, key, ev = judge_idiom(q, r)
        else:
            continue
        if p is None:
            stats[tier][1] += 1
            print(f"ABSTAIN[{tier}] {i} {how} :: {key}")
            continue
        stats[tier][0] += 1
        flip = (i in CT and p != CT[i])
        stats[tier][2] += flip
        rec = {"i": i, "pred": p, "tier": tier, "key": key, "how": how}
        if ev:
            rec["gloss"] = f"[{ev[0]}:{ev[1]}] {ev[2]}"
        out.append(rec)
        print(f"{i} [{tier}] pred={p} think={CT.get(i)} {'FLIP' if flip else 'same'} "
              f":: {key} -> {how}")

    for t, (s, a, f) in stats.items():
        print(f"\n{t}: shipped {s}, abstained {a}, flips vs ctx_think {f}")
    print("pred dist:", {v: sum(1 for o in out if o["pred"] == v) for v in (0, 1)})
    json.dump(out, open(os.path.join(HERE, "source_match_ctx_idiom_upasarga.json"), "w"),
              ensure_ascii=False, indent=1)
