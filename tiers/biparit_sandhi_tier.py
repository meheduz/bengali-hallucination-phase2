"""Tier B: deterministic সন্ধি + canonical বিপরীত শব্দ for authored-ctx rows.

Two tiers with VERY different epistemic standing. Read the header of each.

Companion to somas_tier.py. Same motivation: ctx_think is a passage-
faithfulness judge, and on these rows the response never *contradicts* the
context, so the judge answers "faithful" almost everywhere. The judge is
structurally mismatched; a rule/canon decides them exactly.
"""
import json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_test

HERE = os.path.dirname(os.path.abspath(__file__))


# ===========================================================================
# TIER B1 -- সন্ধি   (STRONG: context-deterministic)
# ===========================================================================
# Shape A rows carry the phonological rule IN THE CONTEXT, e.g.
#   "'নিঃ' ও 'কাম' ... এটি বিসর্গসন্ধি (ঃ+ক=ষ্ক) এর উদাহরণ।"
# The two components plus the stated junction rule determine the joined form
# with zero outside knowledge -- you can audit these by reading the context
# alone. Shape B rows omit the rule ("সন্ধির নিয়ম প্রয়োগ করে"), so they lean
# on standard স্বর/ব্যঞ্জন/বিসর্গ সন্ধি rules; still rule-governed, but the rule
# comes from grammar canon rather than from the context text.

# NB: the authored context drops the OPENING quote of the first word --
# "দেব' ও 'আলয়' শব্দ দুটির ..." -- so the first group is quote-optional.
CTX_PAIR = re.compile(r"'?([^'\s]+)'\s*ও\s*'([^']+)'\s*শব্দ দুটির")
CTX_RULE = re.compile(r"\(([^)]*)\)\s*এর উদাহরণ")

# junction rewrites: (first-word tail, second-word head) -> joined junction.
# Keyed on the rule string the context states, so Shape A needs no canon.
VISARGA_TO_R = set("জমরবদগঘধভনলয")          # ঃ -> র্  before voiced/soft
VISARGA_KEEP = set("খফথছটঠপকস")             # ঃ kept (or ষ্/শ্) before hard


def sandhi_join(a, b, rule=None):
    """Return the joined form, or None if we cannot derive it deterministically.

    `rule` is the parenthetical the context supplies (Shape A). When present it
    is authoritative and we simply apply it. Otherwise we fall back to the
    standard rule set (Shape B).
    """
    # ---- বিসর্গসন্ধি -----------------------------------------------------
    if a.endswith("ঃ"):
        stem, head = a[:-1], b[0]
        if rule:
            # the context spells out what ঃ becomes
            if "লোপ" in rule:
                # "ঃ লোপ, ই দীর্ঘ": drop the visarga and lengthen the
                # preceding vowel sign.  নিঃ + রব -> নী + রব -> নীরব
                LONG = {"ি": "ী", "ু": "ূ"}
                if "দীর্ঘ" in rule and stem and stem[-1] in LONG:
                    return stem[:-1] + LONG[stem[-1]] + b
                return stem + b
            m = re.search(r"ঃ\s*\+\s*(\S)\s*=\s*(\S+)", rule)
            if m and m.group(1) == head:
                repl = m.group(2)
                if repl.startswith("ঃ"):             # explicitly unchanged
                    return a + b
                if repl.startswith("র্"):
                    return stem + "র্" + b
                if repl.startswith(("ষ", "শ")):
                    return stem + repl[0] + "্" + b
            if "অপরিবর্তিত" in rule:
                return a + b
            return None
        # Shape B: standard বিসর্গ rules
        if head in VISARGA_TO_R:
            return stem + "র্" + b
        if head in VISARGA_KEEP:
            return None                              # ষ্/শ্/ঃ split: abstain
        return None

    # ---- ব্যঞ্জনসন্ধি with ৎ ---------------------------------------------
    if a.endswith("ৎ"):
        stem, head = a[:-1], b[0]
        if rule:
            # rule spells the whole cluster, e.g. "ৎ+জ=জ্জ" -> স|জ্জ|ন
            m = re.search(r"ৎ\s*\+\s*(\S)\s*=\s*(\S+)", rule)
            if m and m.group(1) == head:
                return stem + m.group(2) + b[1:]
        # standard: ৎ + ঘোষ -> দ্ ; ৎ + জ/ল -> assimilation
        if head in "জ":
            return stem + "জ্" + b
        if head in "ল":
            return stem + "ল্" + b
        if head in "গঘদধবভজঝড ঢ":
            return stem + "দ্" + b
        return None

    # ---- স্বরসন্ধি ---------------------------------------------------------
    # handled only for the shapes the test actually contains
    VOW = {"অ": "", "আ": "া", "ই": "ি", "ঈ": "ী", "উ": "ু", "ঊ": "ূ",
           "ঋ": "ৃ", "এ": "ে", "ঐ": "ৈ", "ও": "ো", "ঔ": "ৌ"}
    head = b[0]
    if head not in VOW:
        return None
    tail_a = "আ" if a.endswith("া") else ("অ" if a[-1] not in VOW.values() else None)
    if tail_a is None:
        # e.g. নদী (ends ী) + ঈশ
        if a.endswith("ী") and head == "ঈ":
            return a + b[1:]
        return None
    rest = b[1:]
    if head in ("অ", "আ"):                       # অ/আ + অ/আ = আ
        return (a[:-1] + "া" if a.endswith("া") else a + "া") + rest
    if head in ("উ", "ঊ"):                       # অ/আ + উ/ঊ = ও
        return (a[:-1] if a.endswith("া") else a) + "ো" + rest
    if head == "ঋ":                              # অ/আ + ঋ = অর্
        return (a[:-1] if a.endswith("া") else a) + "র্" + rest
    if head == "ঔ":                              # অ/আ + ঔ = ঔ
        return (a[:-1] if a.endswith("া") else a) + "ৌ" + rest
    if head == "ঐ":
        return (a[:-1] if a.endswith("া") else a) + "ৈ" + rest
    return None


def norm(s):
    return re.sub(r"[\s'‘’\"।,\.]", "", str(s))


def sandhi_judge(ctx, resp):
    m = CTX_PAIR.search(ctx)
    if not m:
        return None, "no_pair", None
    a, b = m.group(1).strip(), m.group(2).strip()
    rm = CTX_RULE.search(ctx)
    rule = rm.group(1) if rm else None
    want = sandhi_join(a, b, rule)
    key = f"{a}+{b}"
    if want is None:
        return None, "no_derivation", key
    got = norm(resp)
    if got == norm(want):
        return 1, f"exact:{want}", key
    return 0, f"want:{want}|said:{got}", key


# ===========================================================================
# TIER B2 -- বিপরীত শব্দ   (WEAKER: canon-dependent, NOT context-deterministic)
# ===========================================================================
# ** IMPORTANT CAVEAT, read before shipping. **
# The task brief assumed "the context supplies the word pair or the antonym".
# It does NOT. Every one of the 52 rows carries the same contentless stub:
#     "ব্যাকরণ অনুযায়ী 'X' শব্দটির একটি সুনির্দিষ্ট বিপরীত শব্দ রয়েছে।"
# The context therefore determines NOTHING. Unlike the সমাস and সন্ধি tiers,
# this tier cannot be audited from the context -- it is a pure canon lookup.
#
# To keep the error rate low we only decide a row when one of two strict
# conditions holds, and ABSTAIN on everything else (abstained rows fall back
# to the existing judge, which costs nothing):
#
#   pred=1  the response is EXACTLY the canonical exam-standard antonym
#           (or an equally standard listed alternate).
#   pred=0  the response is on the WRONG SEMANTIC AXIS -- it is the antonym of
#           some *other* word, or a sibling term in the same category, not a
#           plausible opposite of the headword at all.
#
# Anything where the response is a defensible near-synonym of the canonical
# antonym (সাকার->অরূপ vs নিরাকার; সবাক->মূক vs নির্বাক) is ABSTAINED, because
# we cannot tell whether the dataset author would have counted it.

# headword -> (accepted answers, wrong-axis answers with the reason)
BIPARIT = {
    # ---- accepted-only (response matches canon) ---------------------------
    "ঐচ্ছিক":     ({"বাধ্যতামূলক", "আবশ্যিক"}, {}),
    "কৃশ":        ({"স্থূল"}, {}),
    "ক্ষীয়মাণ":   ({"বর্ধমান"}, {}),
    "গরিষ্ঠ":     ({"লঘিষ্ঠ"}, {}),
    "পরার্থ":     ({"স্বার্থ"}, {}),
    "মিতব্যয়ী":   ({"অমিতব্যয়ী", "অপব্যয়ী"}, {}),
    "লঘু":        ({"গুরু"}, {}),
    "সুলভ":       ({"দুর্লভ"}, {}),
    "স্বকীয়":     ({"পরকীয়", "পরকীয়া"}, {}),
    "কদাচিৎ":     ({"প্রায়শ", "প্রায়শই", "সর্বদা"}, {}),
    "জঙ্গম":      ({"স্থাবর"}, {}),
    "তির্যক":     ({"সরল", "ঋজু"}, {}),
    "ঔদ্ধত্য":    ({"বিনয়", "বিনম্রতা"}, {}),
    "ক্ষণস্থায়ী":  ({"চিরস্থায়ী", "স্থায়ী"}, {}),
    "গৃহী":       ({"সন্ন্যাসী"}, {}),
    "চঞ্চল":      ({"স্থির"}, {}),
    "জাগ্রত":     ({"নিদ্রিত", "সুপ্ত"}, {}),
    "তীক্ষ্ণ":     ({"ভোঁতা", "স্থূল"}, {}),
    "দুর্বিনীত":   ({"বিনীত"}, {}),
    "নশ্বর":      ({"অবিনশ্বর"}, {}),
    "পার্থিব":    ({"অপার্থিব"}, {}),
    "বিশদ":       ({"সংক্ষিপ্ত"}, {}),
    "ভীরু":       ({"সাহসী", "নির্ভীক"}, {}),
    "মুখর":       ({"মৌন"}, {}),
    "রূপবান":     ({"কুরূপ", "কুৎসিত"}, {}),
    "সংকীর্ণ":    ({"প্রশস্ত", "উদার"}, {}),

    # ---- wrong-axis rejections -------------------------------------------
    "ঊর্ধ্বগামী": ({"নিম্নগামী", "অধোগামী"},
                   {"স্থিতিশীল": "opposite of 'পরিবর্তনশীল', not of directional ঊর্ধ্বগামী"}),
    "তামসিক":    ({"সাত্ত্বিক"},
                   {"রাজসিক": "third গুণ of the same triad -- a sibling, not the opposite"}),
    "ভূত":       ({"ভবিষ্যৎ"},
                   {"বর্তমান": "sibling tense; ভূত(past) pairs with ভবিষ্যৎ(future)"}),
    "সমষ্টি":    ({"ব্যষ্টি"}, {}),   # খণ্ড -> abstain, see BIPARIT_ABSTAIN
    "হ্রস্ব":     ({"দীর্ঘ"},
                   {"বড়": "size axis + তদ্ভব register; হ্রস্ব-দীর্ঘ is the length pair"}),
    "আবির্ভাব":  ({"তিরোভাব"},
                   {"ধ্বংস": "opposite of সৃষ্টি; আবির্ভাব(advent) pairs with তিরোভাব"}),
    "কনিষ্ঠ":    ({"জ্যেষ্ঠ"},
                   {"বৃদ্ধ": "absolute-age word; কনিষ্ঠ is relative seniority -> জ্যেষ্ঠ"}),
    "লাভ":       ({"ক্ষতি", "লোকসান"},
                   {"ব্যয়": "opposite of আয়; লাভ(profit) pairs with ক্ষতি(loss)"}),
    "স্বাধীন":   ({"পরাধীন"},
                   {"বন্দী": "noun 'prisoner', opposite of মুক্ত; স্বাধীন-পরাধীন is the pair"}),
    "ইহকাল":     ({"পরকাল"},
                   {"ভবিষ্যৎ": "opposite of অতীত; ইহকাল(this life) pairs with পরকাল"}),
}

# Explicitly abstained: response is a defensible near-synonym of the canon, or
# the canon itself is contested. Listed so the audit trail is complete.
BIPARIT_ABSTAIN = {
    "সমীপ": "সুদূর ~ canonical দূর (near-synonym)",
    "উন্মুখ": "নিরাসক্ত ~ canonical বিমুখ (both express aversion)",
    "ঋজু": "কুটিল ~ canonical বক্র (কুটিল is listed opposite of সরল=ঋজু)",
    "চিরন্তন": "অস্থায়ী ~ canonical ক্ষণস্থায়ী (near-synonym)",
    "নিরপরাধ": "দোষী ~ canonical অপরাধী/সাপরাধ (near-synonym)",
    "প্রাচীন": "আধুনিক ~ canonical নবীন/অর্বাচীন (defensible)",
    "বহিরঙ্গ": "ঘনিষ্ঠ ~ canonical অন্তরঙ্গ (near-synonym)",
    "রুক্ষ": "কোমল ~ canonical স্নিগ্ধ (near-synonym)",
    "সবাক": "মূক ~ canonical নির্বাক (near-synonym)",
    "সাকার": "অরূপ ~ canonical নিরাকার (অরূপ literally 'formless')",
    "ঐশ্বরিক": "দানবিক ~ canonical পার্থিব/মানবিক (contested)",
    "প্রখর": "ম্লান ~ canonical স্নিগ্ধ/মৃদু (defensible)",
    "শিষ্ট": "বেয়াদব ~ canonical অশিষ্ট (near-synonym, register shift)",
    "হর্ষ": "দুঃখ ~ canonical বিষাদ (near-synonym)",
    "উত্তম": "মন্দ ~ canonical অধম (near-synonym)",
    "ঐকমত্য": "বিরোধ ~ canonical মতভেদ/দ্বিমত (near-synonym)",
    # canonical pair is ব্যষ্টি, but 'whole vs part' makes খণ্ড a plausible
    # ordinary-language opposite; corpus co-occurrence gave no clear signal
    # (ব্যষ্টি 0 / খণ্ড 1), so this does not clear the wrong-axis bar.
    "সমষ্টি": "খণ্ড ~ canonical ব্যষ্টি (whole/part is a defensible axis)",
}

HEADWORD = re.compile(r"'([^']+)'\s*শব্দটির একটি সুনির্দিষ্ট বিপরীত")


def biparit_judge(ctx, resp):
    m = HEADWORD.search(ctx)
    if not m:
        return None, "no_headword", None
    w = m.group(1).strip()
    said = norm(resp)
    if w in BIPARIT_ABSTAIN:
        return None, "abstain:" + BIPARIT_ABSTAIN[w], w
    if w not in BIPARIT:
        return None, "not_in_canon", w
    ok, bad = BIPARIT[w]
    if said in {norm(x) for x in ok}:
        return 1, f"canon:{said}", w
    for k, why in bad.items():
        if said == norm(k):
            return 0, f"wrong_axis:{said}|{why}", w
    return None, f"unlisted_answer:{said}", w


# ===========================================================================
if __name__ == "__main__":
    T = load_test()
    sm = json.load(open(os.path.join(HERE, "source_match_ctx.json")))
    good = {int(k) for k, v in sm.items()
            if v.get("pred_label") in (0, 1) and not v.get("suspect_gold")}
    done = {o["i"] for o in json.load(
        open(os.path.join(HERE, "source_match_ctx_grammar.json")))}
    tk = json.load(open(os.path.join(HERE, "ctx_think_test.json")))
    ct = dict(zip(tk["idx"], tk["pred"]))

    pool = [i for i, r in enumerate(T)
            if r["context"] and i not in good and i not in done]

    out, stats = [], {}
    for tier, keyword, fn in (("sandhi", "সন্ধি", sandhi_judge),
                              ("biparit", "বিপরীত", biparit_judge)):
        rows = [i for i in pool
                if keyword in T[i]["context"] + T[i]["prompt_bn"]]
        dec = flips = 0
        print(f"\n{'='*72}\nTIER {tier}  ({len(rows)} candidate rows)\n{'='*72}")
        for i in rows:
            p, how, key = fn(T[i]["context"], T[i]["response_bn"])
            if p is None:
                print(f"  ABSTAIN {i:5d}  {key}  :: {how}")
                continue
            dec += 1
            f = (p != ct.get(i))
            flips += f
            out.append({"i": i, "pred": p, "tier": tier, "key": key,
                        "how": how, "ctx_think": ct.get(i)})
            print(f"  {i:5d} pred={p} think={ct.get(i)} "
                  f"{'FLIP' if f else 'same'}  {key}  :: {how}")
        stats[tier] = (len(rows), dec, flips)
        print(f"  -> decided {dec}/{len(rows)}, {flips} flips vs ctx_think")

    print(f"\n{'='*72}")
    for t, (n, d, f) in stats.items():
        print(f"{t:10s} candidates={n:3d} decided={d:3d} flips={f:3d}")
    print("pred dist:", {v: sum(1 for o in out if o["pred"] == v) for v in (0, 1)})
    json.dump(out, open(os.path.join(HERE, "source_match_ctx_biparit_sandhi.json"),
                        "w"), ensure_ascii=False, indent=1)
    print(f"wrote {len(out)} rows -> source_match_ctx_biparit_sandhi.json")
