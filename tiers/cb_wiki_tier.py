"""Wiki-article grounding tier for unsourced closed-book rows 318-380 (clusters C+D).

Mechanism (same family as the validated ctx gold rule): treat the named
bn.wikipedia article as CONTEXT; response containment => faithful.
Because these factoid rows are often *swapped-fact pairs* (the wrong answer
still occurs elsewhere in the same article), whole-article containment is
refined to question-matched sentence windows:

tiers (first applicable wins):
  T1 lead birth/death range: biography lead "(<birth> - <death>)"; birth/death
     year questions checked positionally (swap-safe).
  T2 quoted-title publication year: question quotes a work title + asks
     প্রকাশ-year; article pattern "Title (YYYY)" gives the associated year.
  T3 count adjacency: "কয়টি/কতটি X" questions; number directly adjacent to
     noun X must equal response number (catches 3-খণ্ড/35-পরিচ্ছেদ swaps).
     No adjacency + response number absent from article => ABSTAIN.
  T4 strict window: response numbers+months (or name string) contained in the
     top IDF-scored question-matched sentence (ties allowed).
  T5 fallbacks: multi-component date split across near-duplicate top-3
     sentences; single-year with number-less top sentence -> best sentence of
     the rarest matched stem with directional (সালে-marker) association;
     names -> rarest-stem sentence containment.
Number checks run on canon_numbers() text (norm_v2 strips the dot inside
#6.15#, so it is only used for name containment).
No labels fitted; parameters are structural.
"""
import json, math, os, re, sys, unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bn_num import canon_numbers, norm_v2

BASE = os.path.dirname(os.path.abspath(__file__))
WIKI = os.path.join(BASE, "wiki_articles")

# ---------------- row -> article map (entity named in the question) --------
ROW_ARTICLE = {}
def _span(a, b, art):
    for i in range(a, b + 1):
        ROW_ARTICLE[i] = art
_span(318, 320, "কাজী নজরুল ইসলাম")
_span(321, 324, "জগদীশ চন্দ্র বসু")
_span(325, 327, "পদ্মা সেতু")
_span(328, 331, "সুন্দরবন")
_span(332, 333, "সাকিব আল হাসান")
_span(334, 335, "ঢাকা মেট্রোরেল")
_span(336, 340, "সত্যেন্দ্রনাথ বসু")
_span(341, 342, "শহীদ মিনার_ ঢাকা")
_span(343, 344, "জাতীয় সংসদ ভবন")
_span(345, 345, "কক্সবাজার সমুদ্র সৈকত+ইনানী সমুদ্র সৈকত")
_span(346, 346, "ইনানী সমুদ্র সৈকত+কক্সবাজার সমুদ্র সৈকত")
_span(347, 347, "কক্সবাজার সমুদ্র সৈকত+ইনানী সমুদ্র সৈকত")
_span(348, 350, "লালবাগের কেল্লা")
_span(351, 354, "আহসান মঞ্জিল")
_span(355, 356, "দেবদাস (উপন্যাস)")
_span(357, 357, "আনন্দমঠ")
_span(358, 358, "বন্দে মাতরম্‌+আনন্দমঠ")
_span(359, 360, "পথের পাঁচালী")
_span(361, 362, "গোরা (উপন্যাস)")
_span(363, 364, "কপালকুণ্ডলা")
_span(365, 366, "শেষের কবিতা")
_span(367, 367, "বিষবৃক্ষ (উপন্যাস)")
_span(368, 369, "রাজসিংহ (উপন্যাস)")
_span(370, 370, "মেঘনাদবধ কাব্য")
_span(371, 376, "শরৎচন্দ্র চট্টোপাধ্যায়")
_span(377, 378, "দুর্গেশনন্দিনী")
_span(379, 380, "কৃষ্ণকান্তের উইল")

_cache = {}
def article_text(name):
    if name not in _cache:
        parts = []
        for p in name.split("+"):
            with open(os.path.join(WIKI, p + ".txt")) as f:
                parts.append(f.read())
        _cache[name] = "\n".join(parts)
    return _cache[name]

# ---------------- sentence machinery ----------------------------------------
SENT_SPLIT = re.compile(r"[।৷!?\n]+")

def sentences(text):
    return [s.strip() for s in SENT_SPLIT.split(text) if len(s.strip()) >= 8]

STOP = set("""কে কি কী কোন কোনটি কত কয় কয়টি কতটি কতগুলো কতগুলি কতজন কবে কখন
কোথায় কেন কিভাবে কীভাবে কার কারা কাকে এর ের টি টা জন এবং ও বা হয় হন ছিল ছিলেন
করে করা করেন হয়েছিল হয়েছে হয়েছিলেন সালে সাল সালের খ্রিস্টাব্দে খ্রিষ্টাব্দে বছর বছরের
মোট এই সে তার তাদের তিনি তাকে যে জন্য থেকে দ্বারা মধ্যে অনুযায়ী হিসেবে হিসাবে বলে
জানা যায় রয়েছে আছে দেন পান লাভ গ্রহণ পরে পূর্বে অংশ অংশে মতো নিয়ে দিয়ে হলে
কিলোমিটার মিটার বর্গ টাকা""".split())

ALIAS = {  # derivational forms the prefix rule cannot bridge
    "দৈর্ঘ্য": ["দীর্ঘ"], "প্রশস্ত": ["প্রস্থ", "চওড়া", "প্রশস্ত"],
    "জন্মগ্রহণ": ["জন্ম"], "মৃত্যু": ["মারা", "মৃত্যু", "মৃত্যুবরণ", "প্রয়াত"],
    "প্রকাশিত": ["প্রকাশ"], "উদ্বোধন": ["উদ্বোধ"],
    "রচনা": ["রচিত", "রচে", "লেখেন", "লিখেন"],
}

SUFFIXES = ["য়ের", "ের", "য়ে", "তে", "টির", "টি", "টা", "কে", "রা", "র", "ে"]
TOK = re.compile(r"[ঀ-৿]+|[A-Za-z0-9]+")

def stem(t):
    for _ in range(2):
        for suf in SUFFIXES:
            if t.endswith(suf) and len(t) - len(suf) >= 3:
                t = t[: -len(suf)]
                break
    return t

def qstems(q):
    q = unicodedata.normalize("NFC", str(q))
    out = []
    for t in TOK.findall(q):
        if t in STOP or len(t) < 2:
            continue
        s = stem(t)
        if len(s) >= 3 and s not in STOP:
            out.append(s)
    return list(dict.fromkeys(out))

def stem_in_sentence(s, sent_toks, sent_raw):
    for alias in [s] + ALIAS.get(s, []):
        if len(alias) >= 5 and alias in sent_raw:
            return True
        for t in sent_toks:
            if len(t) >= 3 and (t.startswith(alias) or alias.startswith(t)) \
               and min(len(t), len(alias)) >= max(3, min(len(alias), 4)):
                return True
    return False

MONTHS = ["জানুয়ারি", "ফেব্রুয়ারি", "মার্চ", "এপ্রিল", "মে", "জুন", "জুলাই",
          "আগস্ট", "অগাস্ট", "সেপ্টেম্বর", "অক্টোবর", "নভেম্বর", "ডিসেম্বর"]

def resp_months(resp):
    r = unicodedata.normalize("NFC", str(resp))
    toks = set(TOK.findall(r))
    return [m for m in MONTHS if m in toks]

def month_in(m, sent):
    pat = r"(?<![ঀ-৿])" + re.escape(m) + r"(?![ঀ-৿])"
    return bool(re.search(pat, unicodedata.normalize("NFC", sent)))

NUMPAT = re.compile(r"#(\d+(?:\.\d+)?)#")

def canon(text):
    return canon_numbers(unicodedata.normalize("NFC", str(text)))

def nums(text):
    return set(NUMPAT.findall(canon(text)))

def num_in(n, sent):
    return f"#{n}#" in canon(sent)

COUNT_NOUN = re.compile(r"(?:কতটি|কয়টি|কয়টি|কতগুলো|কতগুলি|কতজন)\s+([ঀ-৿]+)")

def count_noun(q):
    q = unicodedata.normalize("NFC", str(q))
    m = COUNT_NOUN.search(q)
    if not m:
        return None
    n = stem(m.group(1))
    if n in STOP or n in ("সাল", "বছর", "খ্রিস্টাব্দ", "খ্রিষ্টাব্দ"):
        return None
    return n

PUNCT_STRIP = re.compile(r'[।৷,\.\-‐-―\'"“”‘’()!?;:\s]')

# ---------------- scoring ----------------------------------------------------
def rank_sentences(article, question):
    sents = sentences(article)
    stoks = [TOK.findall(unicodedata.normalize("NFC", s)) for s in sents]
    qs = qstems(question)
    if not qs:
        return [], qs, {}, {}
    idf, hit = {}, {}
    N = len(sents)
    for s in qs:
        rows = [i for i in range(N) if stem_in_sentence(s, stoks[i], sents[i])]
        idf[s] = math.log((N + 1) / (len(rows) + 1)) + 0.1
        hit[s] = rows
    scored = []
    for i in range(N):
        sc = sum(idf[s] for s in qs if i in hit[s])
        if sc > 0:
            scored.append((sc, i))
    scored.sort(key=lambda x: (-x[0], x[1]))
    ranked = [(sc, sents[i], i) for sc, i in scored]
    return ranked, qs, idf, hit

BIRTH_Q = re.compile(r"জন্ম")
DEATH_Q = re.compile(r"মৃত্যু|মারা যান|প্রয়াত")
BN2A = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
RANGE = re.compile(r"\((?:[^()]*?)(\d{4})[^()]*?[–—−-][^()]*?(\d{4})")

def lead_range(article):
    head = " ".join(sentences(article)[:3]).translate(BN2A)
    m = RANGE.search(head)
    return (m.group(1), m.group(2)) if m else None

QUOTED = re.compile(r"['‘\"“]([^'’\"”]{2,40})['’\"”]")
PUB_Q = re.compile(r"প্রকাশ")

def title_year(question, article):
    """years adjacent to a quoted work title: Title (YYYY)"""
    q = unicodedata.normalize("NFC", str(question))
    if not PUB_Q.search(q):
        return None
    m = QUOTED.search(q)
    if not m:
        return None
    t = PUNCT_STRIP.sub("", m.group(1))
    if len(t) < 3:
        return None
    art = PUNCT_STRIP.sub("", canon(article))
    years = re.findall(re.escape(t) + r"[ঀ-৿]{0,3}#(\d{4})#", art)
    return years or None

def classify(question, response, article):
    """returns (pred, diag); pred may be None => abstain"""
    ranked, qs, idf, _ = rank_sentences(article, question)
    art_canon = canon(article)
    resp_norm = norm_v2(response)
    rnum = nums(response) - nums(question)
    rmon = resp_months(response)
    whole = dict(
        resp_in_article=resp_norm in norm_v2(article) if resp_norm else False,
        nums_in_article=all(f"#{n}#" in art_canon for n in rnum) if rnum else None,
    )
    qn = unicodedata.normalize("NFC", str(question))
    d = dict(rnum=sorted(rnum), rmon=rmon, **whole,
             top=[(round(sc, 2), s[:130]) for sc, s, _ in ranked[:3]])

    # T1: biography lead (birth - death) range, single-year responses
    years = [n for n in rnum if len(n) == 4]
    if len(rnum) == 1 and len(years) == 1 and not rmon:
        rng = lead_range(article)
        if rng:
            if BIRTH_Q.search(qn) and not DEATH_Q.search(qn):
                return int(years[0] == rng[0]), dict(tier="lead_birth", rng=rng, **d)
            if DEATH_Q.search(qn) and not BIRTH_Q.search(qn):
                return int(years[0] == rng[1]), dict(tier="lead_death", rng=rng, **d)

    # T2: quoted-title publication year adjacency
    if len(years) == 1 and len(rnum) == 1:
        ty = title_year(question, article)
        if ty:
            return int(years[0] in ty), dict(tier="title_year", ty=ty, **d)

    if not ranked:
        pred = 1 if (whole["resp_in_article"] or whole["nums_in_article"]) else 0
        return pred, dict(tier="whole_article_fallback", **d)

    topscore = ranked[0][0]
    cands = [s for sc, s, _ in ranked if sc >= topscore - 1e-9][:3]
    top3 = [s for _, s, _ in ranked[:3]]

    # T3: count-noun adjacency
    cn = count_noun(question)
    if cn is not None and rnum:
        adj = []
        for sent in top3:
            cs = PUNCT_STRIP.sub("", canon(sent))
            for m in re.finditer(r"#(\d+)#[ঀ-৿]{0,6}?" + re.escape(cn), cs):
                adj.append(m.group(1))
        d.update(cn=cn, adj=adj)
        if adj:
            return int(any(v in rnum for v in adj)), dict(tier="count_adjacency", **d)
        if not whole["nums_in_article"]:
            return None, dict(tier="count_abstain", **d)  # fact absent from article

    def sent_ok(sent):
        if rnum:
            return all(num_in(n, sent) for n in rnum) and \
                   all(month_in(m, sent) for m in rmon)
        return bool(resp_norm) and resp_norm in norm_v2(sent)

    # T4: strict window (tie-top sentences)
    if any(sent_ok(s) for s in cands):
        return 1, dict(tier="window_top1", **d)

    # T5a: multi-component date split across top-3 near-duplicate sentences
    comps = len(rnum) + len(rmon)
    if comps >= 2:
        found_all = all(any(num_in(n, s) for s in top3) for n in rnum) and \
                    all(any(month_in(m, s) for s in top3) for m in rmon)
        co2 = any(sum([num_in(n, s) for n in rnum] +
                      [month_in(m, s) for m in rmon]) >= 2 for s in top3)
        if found_all and co2:
            return 1, dict(tier="window_multi3", **d)

    # T5b: single-year responses whose tie-top evidence sentence has no
    # year-like number at all (entity-heavy lead crowded out the fact
    # sentence). Accept iff some sentence contains the response year as its
    # ONLY 4-digit year, all response months, and >=2 matched question stems.
    # Swapped-fact sentences carry both years and are excluded by design.
    YEARISH = re.compile(r"#\d{3,4}#")
    if len(rnum) == 1 and len(years) == 1 and \
       not any(YEARISH.search(canon(s)) for s in cands):
        y = years[0]
        for _, sent, _ in ranked:
            cs = canon(sent)
            ys = set(re.findall(r"#(\d{4})#", cs))
            ys = {v for v in ys if 1200 <= int(v) <= 2100}
            if ys != {y}:
                continue
            if not all(month_in(m, sent) for m in rmon):
                continue
            stoksent = TOK.findall(unicodedata.normalize("NFC", sent))
            hits = [s for s in qs if stem_in_sentence(s, stoksent, sent)]
            if len(hits) >= 2:
                d.update(sy_sent=sent[:130], sy_stems=hits)
                return 1, dict(tier="single_year_sent", **d)

    # T5c: name responses -> containment in the rarest matched stem's best
    # sentence (kept at top-1 stem to bound swap risk)
    rare_sent = None
    ms = sorted([(idf[s], s) for s in qs], reverse=True)
    for _, s in ms:
        rows = [(sc, sent) for sc, sent, i in ranked
                if stem_in_sentence(s, TOK.findall(unicodedata.normalize("NFC", sent)), sent)]
        if rows:
            rare_sent = max(rows)[1]
            d["rare_stem"] = s
            break
    d["rare_sent"] = rare_sent[:130] if rare_sent else None
    if rare_sent and not rnum and resp_norm and resp_norm in norm_v2(rare_sent):
        return 1, dict(tier="rare_stem_name", **d)

    return 0, dict(tier="window_top1", **d)

# ---------------- run ---------------------------------------------------------
if __name__ == "__main__":
    from common import load_test, load_samples
    rows = load_test()
    idx = json.load(open(os.path.join(BASE, "cb379_idx.json")))
    targets = [i for i in idx if i in ROW_ARTICLE]

    out, diags, abstain = [], [], []
    for i in targets:
        r = rows[i]
        art = ROW_ARTICLE[i]
        pred, d = classify(r["prompt_bn"], r["response_bn"], article_text(art))
        if pred is None:
            abstain.append(i)
        else:
            out.append(dict(i=i, pred=pred, article=art.split("+")[0]))
        diags.append((i, r, art, pred, d))

    for i, r, art, pred, d in diags:
        print(f"== {i} [{art.split('+')[0]}] pred={pred} tier={d['tier']}")
        print(f"   Q: {r['prompt_bn'][:100]}")
        print(f"   R: {r['response_bn'][:80]}  nums={d.get('rnum')} mon={d.get('rmon')} "
              f"cn={d.get('cn')} adj={d.get('adj')} rng={d.get('rng')} ty={d.get('ty')}")
        if d.get("rare_sent"):
            print(f"   rare[{d.get('rare_stem')}]: {d['rare_sent']}")
        if d.get("assoc_stems") is not None:
            print(f"   assoc: {d.get('assoc_stems')}")
        for sc, s in d.get("top", [])[:2]:
            print(f"   [{sc}] {s}")
        print()

    # ---------- validation on labeled cb samples naming these entities ------
    S = load_samples()
    VAL = [
        (2, "বঙ্কিমচন্দ্র চট্টোপাধ্যায়"),   # কাঁঠালপাড়া -> Bankim's birthplace article
        (30, "সুন্দরবন"),
        (105, "সাকিব আল হাসান"),
    ]
    print("==== VALIDATION ====")
    ok = n = 0
    for si, art in VAL:
        r = S[si]
        if r["context"]:
            continue
        pred, d = classify(r["prompt_bn"], r["response_bn"], article_text(art))
        n += 1
        ok += int(pred == r["label"])
        print(f"sample {si} [{art}] label={r['label']} pred={pred} tier={d['tier']} "
              f"{'OK' if pred == r['label'] else 'WRONG'}")
        print(f"   Q: {r['prompt_bn'][:90]}  R: {r['response_bn'][:60]}")
        for sc, s in d.get("top", [])[:1]:
            print(f"   [{sc}] {s}")
    print(f"validation: {ok}/{n}")

    json.dump(out, open(os.path.join(BASE, "source_match_cb_wiki.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"\nwrote {len(out)} rows (abstained: {abstain}); "
          f"pred=1: {sum(o['pred'] for o in out)}, pred=0: {sum(1 - o['pred'] for o in out)}")
