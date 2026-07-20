"""WIKISOURCE / DICTIONARY tier for the unsourced closed-book idiom rows.

Covers the archaic Bengali idiom/dictionary rows (cb379 cluster A1) whose heads
are ABSENT from bn.wiktionary (হাবড়হাটি, এমুড়ো-ওমুড়ো, চালুমাল, ফইজত ...).
Per analysis/cb379_clusters.md they come from বাংলা একাডেমি ব্যবহারিক অভিধান /
জ্ঞানেন্দ্রমোহন দাস "বাঙ্গালা ভাষার অভিধান".

Two retrieval backends, tried in order:
  1. accessibledictionary.gov.bd  — Bangla Academy dictionary DB behind a plain
     PHP endpoint: POST /inc/bn-to-bn-home.php  q=<word>&RadioGroup1=bn-to-bn
     (discovered from the site's own inline JS; clean digital glosses).
  2. bn.wikisource.org            — Das dictionary, proofread Page: namespace
     (ns 104), CirrusSearch exact-phrase then wikitext window around the head.
     OCR is noisy but overlap-scoring only counts response tokens found in it.

Prediction = content-token overlap of the response against the retrieved gloss,
same mechanism/threshold (0.34) as the validated wikt_idioms.py tier.

Run:  python wikisource_tier.py            (validate on samples + predict test)
"""
import json, re, os, sys, time, unicodedata, html
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_samples, load_test
from wikt_idioms import extract_idiom, content_tokens, build_lookup

WORK = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(WORK, "source_data", "wikisource_tier_cache.json")

S = requests.Session()
S.headers["User-Agent"] = "datathon-research/0.1 (meheduz900@gmail.com)"
WS_API = "https://bn.wikisource.org/w/api.php"
GOVBD = "https://accessibledictionary.gov.bd/inc/bn-to-bn-home.php"

cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

def save_cache():
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(cache, open(CACHE, "w"), ensure_ascii=False)

# ---------------------------------------------------------------- normalizing
ZW = "‌‍﻿"
def norm(s):
    s = unicodedata.normalize("NFC", str(s))
    return re.sub(rf"[\s\-—–'\"“”()।;,:.!?{ZW}]+", "", s)

FOLD = str.maketrans({"ূ": "ু", "ী": "ি"})
def fold(s):
    return norm(s).translate(FOLD)

def core_head(head):
    """Strip parenthetical qualifiers: 'ঘাড় ভাঙা (পরের)' -> 'ঘাড় ভাঙা'."""
    return re.sub(r"\([^)]*\)", " ", head).strip()

def word_stems(w):
    out = [w]
    if w.endswith("ের") and len(w) > 3: out.append(w[:-2])
    elif w.endswith("র") and len(w) > 3: out.append(w[:-1])
    elif w.endswith("ে") and len(w) > 3: out.append(w[:-1])   # কপালে -> কপাল
    return out

# ------------------------------------------------------- backend 1: gov.bd BA
LI = re.compile(r"<li>(.*?)</li>", re.S)
# headword = separator span (exact-match layout) or <strong> (sub-entry layout).
# <b> is NOT a headword — it marks the search-hit substring, wherever it fell.
HEADSPAN = re.compile(r'<span class="separator">(.*?)</span>|<strong>(.*?)</strong>', re.S)
TAG = re.compile(r"<[^>]+>")

def govbd_query(q):
    """Return [(headword_part, item_text)] from the DIRECT-match article only
    (the trailing 'Nearby Words' article is fuzzy neighbours — excluded)."""
    key = "govbd:" + q
    if key not in cache:
        try:
            r = S.post(GOVBD, data={"q": q, "RadioGroup1": "bn-to-bn"}, timeout=25)
            cache[key] = r.text
        except requests.RequestException:
            cache[key] = ""
        time.sleep(0.25)
    first = cache[key].split("Nearby Words")[0]
    out = []
    for li in LI.findall(first):
        heads = " , ".join(a or b for a, b in HEADSPAN.findall(li))
        heads = re.sub(r"Bengali Word|Bengali definition", " ", TAG.sub(" ", heads))
        text = re.sub(r"\s+", " ", html.unescape(TAG.sub(" ", li))).strip()
        text = re.sub(r"Bengali Word|Bengali definition", " ", text)
        if text:
            out.append((heads, text))
    return out

def head_matches(headpart, core):
    """True iff some comma-separated headword equals core (homograph digits
    stripped): 'টিকা ২, টীকা ২' matches টিকা; 'গুটিকা' does not."""
    for part in re.split(r"[,;/]", headpart):
        if norm(re.sub(r"[০-৯0-9]+", "", part)) == norm(core):
            return True
    return False

def find_norm_all(hay, needle):
    """All (start, end) raw-text spans where the normalized needle occurs.
    NB: norm(c) of a single char can emit 2 chars (য়/ড়/ঢ় are composition-
    excluded, so NFC *decomposes* them) — index map must track emitted chars."""
    nh, idxmap = [], []
    for i, c in enumerate(hay):
        nc = norm(c)
        nh.append(nc)
        idxmap.extend([i] * len(nc))
    nhs, nn = "".join(nh), norm(needle)
    out, p = [], nhs.find(nn)
    while p >= 0 and nn:
        out.append((idxmap[p], idxmap[p + len(nn) - 1] + 1))
        p = nhs.find(nn, p + 1)
    return out

BNCHAR = re.compile(r"[ঀ-৿]")

def classify_occurrence(text, st, en, core):
    """'own-word' boundary + quotation analysis for a match inside entry text.
    Returns 'reject' | 'example' (inside a short usage-parenthetical, gloss
    precedes it) | 'subentry' (outside parens, gloss follows)."""
    if en < len(text) and BNCHAR.match(text[en]):
        return "reject-word"     # part of a longer word (চরণদাস in চরণদাসী)
    depth = text[:st].count("(") - text[:st].count(")")
    if depth <= 0:
        return "subentry"
    lp = text[:st].rfind("(")
    rp = text.find(")", en)
    par = text[lp + 1:rp if rp >= 0 else len(text)]
    # long parenthetical or one with an attribution = a literary quotation
    if "-(" in par or len(norm(par)) > len(norm(core)) + 12:
        return "reject-quote"
    return "example"             # short usage example: '(কাঁচা সোনা)'

def govbd_lookup(head):
    """Returns (gloss, how, loose_gloss). Strict tiers:
    1 own   — item whose headword equals the head;
    2 sense — direct-match sense line / sub-entry occurrence in an entry text
              (usage-quote occurrences rejected).
    loose_gloss (all head words present in one non-rejected item) is returned
    separately so the caller can put wikisource before it."""
    core = core_head(head)
    words = [w for w in core.split() if len(w) > 1]
    queries = [core, core.replace("-", " ")]
    for w in words:
        queries += word_stems(w)
    own, sub, pool = [], [], []
    for q in dict.fromkeys(queries):
        for heads, text in govbd_query(q):
            if head_matches(heads, core):
                own.append(text); continue
            quote_only = False
            for st, en in find_norm_all(text, core):
                kind = classify_occurrence(text, st, en, core)
                if kind == "reject-quote":
                    quote_only = True; continue
                if kind == "reject-word":
                    continue
                if len(text) <= 350:
                    sub.append(text)
                elif kind == "subentry":
                    # is this a sub-entry HEAD ('কথা ফেলা (ক্রিয়া) গ্লস…') or the
                    # head in synonym position at the END of another entry's
                    # gloss ('…পুনরায় চর্বণ; চর্বিতচর্বণ (quote-…)')? If a long/
                    # attributed parenthetical follows, gloss lies BEHIND it.
                    tail = re.match(r"\s*\(([^)]*)", text[en:])
                    if tail and ("-(" in tail.group(1)
                                 or len(norm(tail.group(1))) > len(norm(core)) + 12):
                        sub.append(text[max(0, st - 250):en])
                    else:
                        sub.append(text[st:st + 250])
                else:
                    sub.append(text[max(0, st - 250):en + 60])
                quote_only = False
                break
            if not quote_only:
                pool.append(text)
        if own:
            return " ; ".join(dict.fromkeys(own)), f"govbd[{q}]", None
        if sub:
            return " ; ".join(dict.fromkeys(sub)), f"govbd-sub[{q}]", None
    # loose: some single (non-quote-rejected) item has every head word
    hit = [i for i in pool
           if all(any(fold(st) in fold(i) for st in word_stems(w)) for w in words)]
    if hit:
        return None, None, " ; ".join(dict.fromkeys(hit))
    return None, None, None

# ---------------------------------------------- backend 2: bn.wikisource Das
def ws_get(params):
    key = "ws:" + json.dumps(params, sort_keys=True, ensure_ascii=False)
    if key not in cache:
        try:
            cache[key] = S.get(WS_API, params={**params, "format": "json"},
                               timeout=25).json()
        except requests.RequestException:
            cache[key] = {}
        time.sleep(0.2)
    return cache[key]

def ws_search(q):
    r = ws_get({"action": "query", "list": "search", "srsearch": q,
                "srnamespace": "104", "srlimit": 10})
    return [h["title"] for h in r.get("query", {}).get("search", [])]

def ws_wikitext(title):
    r = ws_get({"action": "query", "prop": "revisions", "rvprop": "content",
                "rvslots": "main", "titles": title})
    for pg in r.get("query", {}).get("pages", {}).values():
        try:
            return pg["revisions"][0]["slots"]["main"]["*"]
        except (KeyError, IndexError):
            pass
    return ""

def strip_wiki(wt):
    wt = re.sub(r"<noinclude>.*?</noinclude>", " ", wt, flags=re.S)
    wt = re.sub(r"\{\{[^}]*\}\}", " ", wt)
    wt = re.sub(r"<[^>]+>", " ", wt)
    return re.sub(r"\s+", " ", wt)

def quoted_context(text, pos, back=70):
    """True if the occurrence at pos sits inside a usage quotation:
    Das marks examples 'প্র—“...”' — reject unclosed “ or a প্র-dash marker."""
    ctx = text[max(0, pos - back):pos]
    if re.search(r"প্[রৰ]\s*[-—–]", ctx):
        return True
    if "“" in ctx and "”" not in ctx.rsplit("“", 1)[1]:
        return True
    return False

def wikisource_lookup(head, window=700):
    core = core_head(head)
    for v in dict.fromkeys([core, core.replace("-", " "), core.replace(" ", "")]):
        titles = [t for t in ws_search(f'"{v}"') if "ভাষার অভিধান" in t]
        for title in titles[:2]:
            text = strip_wiki(ws_wikitext(title))
            for st, en in find_norm_all(text, v):
                if not quoted_context(text, st):
                    return text[st:st + window], f"wikisource[{title}]"
    return None, None

# ------------------------------------------------------------------- predict
def lookup(head, qtype="শাব্দিক অর্থ"):
    """gov.bd own-entry/sense tiers, then wikisource Das, then gov.bd loose.
    The loose tier joins the component words' entries — valid evidence for a
    LITERAL (শাব্দিক) reading of a multi-word phrase, but figurative (ভাবার্থ)
    meanings are non-compositional, so loose is disallowed there."""
    gloss, src, loose = govbd_lookup(head)
    if gloss is None:
        gloss, src = wikisource_lookup(head)
    if gloss is None and loose is not None:
        if "ভাবার্থ" in qtype and " " in core_head(head).strip():
            return None, None
        gloss, src = loose, "govbd-loose"
    return gloss, src

def _one_edit(a, b):
    la, lb = len(a), len(b)
    if la == lb:
        return sum(x != y for x, y in zip(a, b)) == 1
    if la > lb:
        a, b, la, lb = b, a, lb, la
    if lb - la != 1:
        return False
    i = j = diff = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1; j += 1
        else:
            diff += 1
            if diff > 1:
                return False
            j += 1
    return True

def tok_covered(t, gt):
    """Exact hit, or edit-distance-1 hit for tokens >=4 chars — absorbs the
    dictionary DB's own typos (আত্নসাৎ for আত্মসাৎ) and OCR one-offs."""
    if t in gt:
        return True
    return len(t) >= 4 and any(abs(len(g) - len(t)) <= 1 and _one_edit(t, g)
                               for g in gt)

def predict(gloss, response, thresh=0.34):
    rt = content_tokens(response)
    if not rt:
        return None, "empty-resp"
    gt = content_tokens(gloss)
    ov = sum(tok_covered(t, gt) for t in rt) / len(rt)
    return (1 if ov >= thresh else 0), f"overlap={ov:.2f}"

# ---------------------------------------------------------------------- main
if __name__ == "__main__":
    # ---- validation: labeled sample idiom rows NOT covered by wiktionary
    wikt_lut = build_lookup(json.load(open(os.path.join(WORK, "wikt_pages.json"))))
    def in_wikt(idm):
        return idm in wikt_lut or any(
            t.replace(" ", "") == idm.replace(" ", "") for t in wikt_lut)

    val = []
    for r in load_samples():
        if r["context"]: continue
        idm, qt = extract_idiom(r["prompt_bn"])
        if idm and not in_wikt(idm):
            val.append((idm, qt, r["response_bn"], r["label"]))

    print(f"validation rows (sample idioms absent from wiktionary): {len(val)}")
    ok = bad = nc = 0
    for idm, qt, resp, label in val:
        gloss, src = lookup(idm, qt)
        save_cache()
        if gloss is None:
            nc += 1; print(f"  NOSRC {idm}"); continue
        p, how = predict(gloss, resp)
        if p is None:
            nc += 1; continue
        tag = "ok  " if p == label else "MISS"
        if p == label: ok += 1
        else: bad += 1
        print(f"  {tag} {idm} | pred={p} true={label} {how} | {src}")
        if p != label:
            print(f"        resp:  {str(resp)[:70]}")
            print(f"        gloss: {gloss[:110]}")
    cov = ok + bad
    print(f"validation: covered {cov}/{len(val)}, acc={ok/max(1,cov):.3f}")

    # ---- test predictions
    T = load_test()
    cb_idx = set(json.load(open(os.path.join(WORK, "cb379_idx.json"))))
    wikt_done = {r["i"] for r in json.load(open(os.path.join(WORK, "wikt_test_pred.json")))}
    out = []
    misses = []
    for i in sorted(cb_idx - wikt_done):
        idm, qt = extract_idiom(T[i]["prompt_bn"])
        if not idm: continue
        gloss, src = lookup(idm, qt)
        save_cache()
        if gloss is None:
            misses.append((i, idm, "no-source")); continue
        p, how = predict(gloss, T[i]["response_bn"])
        if p is None:
            misses.append((i, idm, "empty-resp")); continue
        out.append({"i": i, "pred": p, "head": idm, "source": src, "how": how,
                    "gloss": gloss[:400]})
        print(f"test {i:4d} pred={p} {how:14s} {idm} | {src}")
    for i, idm, why in misses:
        print(f"test {i:4d} SKIP ({why}) {idm}")

    with open(os.path.join(WORK, "source_match_cb_wikisource.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\nwrote {len(out)} predictions -> source_match_cb_wikisource.json "
          f"(skipped {len(misses)})")
    print(f"pred distribution: 1={sum(1 for r in out if r['pred']==1)} "
          f"0={sum(1 for r in out if r['pred']==0)}")
