"""bn.wiktionary বাগধারা gold tier.

The idiom-template rows ("X" এর ভাবার্থ/শাব্দিক অর্থ কী?) are sourced from
bn.wiktionary entries whose wikitext has ভাবার্থ / শাব্দিক অর্থ sections.
Parse those sections, match rows by idiom lookup, predict by meaning-overlap.

Validated on labeled sample idiom rows before shipping (gate >= 0.95).
"""
import json, re, collections
from common import load_samples, load_test
from bn_num import norm_v2

def clean(b):
    b = re.sub(r"\{\{[^}]*\}\}", " ", b)
    b = re.sub(r"\[\[([^|\]]*\|)?([^\]]*)\]\]", r"\2", b)
    b = re.sub(r"<[^>]+>", " ", b)
    b = re.sub(r"[*:']+", " ", b)
    return re.sub(r"\s+", " ", b).strip()

def parse_sections(wikitext):
    """Extract the meaning glosses.

    bn.wiktionary puts definitions as '# gloss' lines under a POS header
    (ক্রিয়াপদ / বিশেষ্য / বাগধারা ...), not under a ভাবার্থ header. Collect all
    gloss lines; also keep any explicit ভাবার্থ/শাব্দিক-অর্থ sections when present.
    """
    out = {}
    glosses = [clean(m) for m in re.findall(r"^#\s*([^\n#*:].*)$", wikitext, re.M)]
    glosses = [g for g in glosses if g]
    if glosses:
        out["gloss"] = " ; ".join(glosses)
    parts = re.split(r"==+\s*([^=]+?)\s*==+", wikitext)
    for i in range(1, len(parts) - 1, 2):
        name = parts[i].strip()
        if "ভাবার্থ" in name or "শাব্দিক" in name:
            body = clean(re.sub(r"^#\s*", "", parts[i + 1], flags=re.M))
            if body:
                out[name] = body
    return out

IDIOM_Q = re.compile(r'^[\"“]?(.+?)[\"”]?\s*এর\s*(ভাবার্থ|শাব্দিক অর্থ)\s*কী\s*\??$')

def extract_idiom(prompt):
    m = IDIOM_Q.match(str(prompt).strip())
    if m:
        return m.group(1).strip().strip('"“”'), m.group(2)
    return None, None

def content_tokens(s):
    STOP = {"করা", "হওয়া", "যে", "বা", ";", ","}
    toks = set(re.findall(r"[ঀ-৿]+", str(s)))
    return {t for t in toks if len(t) > 1 and t not in STOP}

def build_lookup(pages):
    lut = {}
    for title, wt in pages.items():
        secs = parse_sections(wt)
        if secs:
            lut[title.strip()] = secs
    return lut

def predict(idiom, qtype, response, lut):
    """Return (pred, how) or (None, reason)."""
    entry = lut.get(idiom)
    if entry is None:
        # try normalized lookup
        for t in lut:
            if t.replace(" ", "") == idiom.replace(" ", ""):
                entry = lut[t]; break
    if entry is None:
        return None, "no-entry"
    # pick the section matching the question type; else any
    texts = [v for k, v in entry.items() if qtype in k] or list(entry.values())
    gold = " ".join(texts)
    rt = content_tokens(response)
    gt = content_tokens(gold)
    if not rt:
        return None, "empty-resp"
    ov = len(rt & gt) / len(rt)
    return (1 if ov >= 0.34 else 0), f"overlap={ov:.2f}"

if __name__ == "__main__":
    pages = json.load(open("wikt_pages.json"))
    lut = build_lookup(pages)
    print(f"wiktionary entries with meaning sections: {len(lut)}")

    # validate on labeled sample idiom rows
    S = load_samples()
    rows = [(r, *extract_idiom(r["prompt_bn"])) for r in S if not r["context"]]
    rows = [(r, idm, qt) for r, idm, qt in rows if idm]
    ok = bad = nc = 0
    for r, idm, qt in rows:
        p, how = predict(idm, qt, r["response_bn"], lut)
        if p is None: nc += 1; continue
        if p == r["label"]: ok += 1
        else:
            bad += 1
            print(f"  MISS: {idm} | {how} | true={r['label']} resp={str(r['response_bn'])[:50]}")
    print(f"sample idiom rows: {len(rows)}, covered {ok+bad}, acc={ok/max(1,ok+bad):.3f}, no-entry {nc}")
