"""Bengali numeral canonicalization for substring matching.

Parameter-free, linguistic: maps number WORDS (এক..নিরানব্বই, শ/শত, হাজার,
লক্ষ/লাখ, কোটি) and Bengali/ASCII digits to a canonical delimited token
#<value>#, so 'আঠারশ বত্রিশ' == '১৮৩২' == '1832'. Applied identically to
response and context; the # delimiters prevent a lone digit matching inside
a longer number (e.g. ৩ inside ১৮৩২).

This is a canonicalization (like the punctuation stripping already in norm()),
not a fitted parameter.
"""
import re

BN2ASCII = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

UNITS = {
    "শূন্য": 0, "এক": 1, "দুই": 2, "দু": 2, "তিন": 3, "চার": 4, "পাঁচ": 5,
    "পাচ": 5, "ছয়": 6, "ছয়": 6, "সাত": 7, "আট": 8, "নয়": 9, "নয়": 9,
    "দশ": 10, "এগারো": 11, "এগার": 12 - 1, "বারো": 12, "বার": 12,
    "তেরো": 13, "তের": 13, "চৌদ্দ": 14, "চোদ্দ": 14, "পনেরো": 15, "পনের": 15,
    "ষোল": 16, "ষোলো": 16, "সতেরো": 17, "সতের": 17, "আঠারো": 18, "আঠার": 18,
    "উনিশ": 19, "ঊনিশ": 19, "বিশ": 20, "কুড়ি": 20, "একুশ": 21, "বাইশ": 22,
    "তেইশ": 23, "চব্বিশ": 24, "পঁচিশ": 25, "পচিশ": 25, "ছাব্বিশ": 26,
    "সাতাশ": 27, "আঠাশ": 28, "আটাশ": 28, "ঊনত্রিশ": 29, "উনত্রিশ": 29,
    "ত্রিশ": 30, "একত্রিশ": 31, "বত্রিশ": 32, "তেত্রিশ": 33, "চৌত্রিশ": 34,
    "পঁয়ত্রিশ": 35, "পঁয়ত্রিশ": 35, "ছত্রিশ": 36, "সাঁইত্রিশ": 37, "আটত্রিশ": 38,
    "ঊনচল্লিশ": 39, "উনচল্লিশ": 39, "চল্লিশ": 40, "একচল্লিশ": 41,
    "বিয়াল্লিশ": 42, "বিয়াল্লিশ": 42, "তেতাল্লিশ": 43, "চুয়াল্লিশ": 44,
    "পঁয়তাল্লিশ": 45, "পঁয়তাল্লিশ": 45, "ছেচল্লিশ": 46, "সাতচল্লিশ": 47,
    "আটচল্লিশ": 48, "ঊনপঞ্চাশ": 49, "উনপঞ্চাশ": 49, "পঞ্চাশ": 50,
    "একান্ন": 51, "বাহান্ন": 52, "তিপ্পান্ন": 53, "চুয়ান্ন": 54, "পঞ্চান্ন": 55,
    "ছাপ্পান্ন": 56, "সাতান্ন": 57, "আটান্ন": 58, "ঊনষাট": 59, "উনষাট": 59,
    "ষাট": 60, "একষট্টি": 61, "বাষট্টি": 62, "তেষট্টি": 63, "চৌষট্টি": 64,
    "পঁয়ষট্টি": 65, "পঁয়ষট্টি": 65, "ছেষট্টি": 66, "সাতষট্টি": 67, "আটষট্টি": 68,
    "ঊনসত্তর": 69, "উনসত্তর": 69, "সত্তর": 70, "একাত্তর": 71, "বাহাত্তর": 72,
    "তিয়াত্তর": 73, "চুয়াত্তর": 74, "পঁচাত্তর": 75, "ছিয়াত্তর": 76,
    "সাতাত্তর": 77, "আটাত্তর": 78, "ঊনআশি": 79, "ঊনাশি": 79, "উনাশি": 79,
    "আশি": 80, "একাশি": 81, "বিরাশি": 82, "তিরাশি": 83, "চুরাশি": 84,
    "পঁচাশি": 85, "ছিয়াশি": 86, "সাতাশি": 87, "আটাশি": 88,
    "ঊননব্বই": 89, "উননব্বই": 89, "নব্বই": 90, "একানব্বই": 91,
    "বিরানব্বই": 92, "তিরানব্বই": 93, "চুরানব্বই": 94, "পঁচানব্বই": 95,
    "ছিয়ানব্বই": 96, "সাতানব্বই": 97, "আটানব্বই": 98, "নিরানব্বই": 99,
}
HUNDRED = {"শ", "শত", "শো"}
BIG = {"হাজার": 1000, "লক্ষ": 100000, "লাখ": 100000, "কোটি": 10000000}
COUNTERS = {"টি", "টা", "জন", "খানা", "খানি"}  # dropped when following a number

SPLIT = re.compile(r"[।,\.\-‐-―'\"“”‘’()!?;:\s]+")


def _tok_value(tok):
    """value of a single token, or None. Handles digits, unit words,
    unit+শ compounds (আঠারশ), unit+counter (সাতটি)."""
    if not tok:
        return None
    t = tok.translate(BN2ASCII)
    if re.fullmatch(r"\d+(?:\.\d+)?", t):
        return float(t) if "." in t else int(t)
    if tok in UNITS:
        return UNITS[tok]
    if tok in HUNDRED:
        return "H"
    if tok in BIG:
        return ("B", BIG[tok])
    # unit + শ/শো  (আঠারশ = 18*100)
    for h in ("শো", "শ"):
        if tok.endswith(h) and tok[:-len(h)] in UNITS:
            return UNITS[tok[:-len(h)]] * 100
    # unit + counter suffix (সাতটি, দুটি)
    for c in COUNTERS:
        if tok.endswith(c) and tok[:-len(c)] in UNITS:
            return UNITS[tok[:-len(c)]]
    return None


def _fmt(v):
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return f"#{v}#"


TOKEN = re.compile(r"\d+(?:\.\d+)?|[^।,\.\-‐-―'\"“”‘’()!?;:\s\d]+")


def _kind(tok):
    """('digit', v) | ('unit', v) | ('H',) | ('B', mult) | None"""
    if re.fullmatch(r"\d+(?:\.\d+)?", tok):
        return ("digit", float(tok) if "." in tok else int(tok))
    if tok in UNITS:
        return ("unit", UNITS[tok])
    if tok in HUNDRED:
        return ("H",)
    if tok in BIG:
        return ("B", BIG[tok])
    for h in ("শো", "শ"):
        if tok.endswith(h) and tok[:-len(h)] in UNITS:
            return ("unit", UNITS[tok[:-len(h)]] * 100)
    for c in COUNTERS:
        if tok.endswith(c) and tok[:-len(c)] in UNITS:
            return ("unit", UNITS[tok[:-len(c)]])
    return None


def canon_numbers(text):
    """Replace every maximal number-word/digit span with #<value>#.

    Digit runs are their own tokens (so ৩৯টি, ১৮৭৬[1] parse); digit-grouping
    commas are merged first. Adjacent digit numbers never merge (১৯৯৪ ১৯৯৫
    stays two numbers); word compounds do (আঠারশ বত্রিশ -> 1832,
    দুই লক্ষ তেরো হাজার -> 213000)."""
    t = str(text).translate(BN2ASCII)
    t = re.sub(r"(?<=\d),(?=\d)", "", t)          # 5,22,000 -> 522000
    tokens = list(TOKEN.finditer(t))
    parts, pos, i = [], 0, 0
    while i < len(tokens):
        k = _kind(tokens[i].group(0))
        if k is None or k[0] in ("H", "B"):       # bare multiplier: leave as text
            i += 1
            continue
        # start of a number span at token i
        total, cur = 0, k[1]
        last = k[0]                               # 'digit' or 'unit'
        j = i + 1
        while j < len(tokens):
            nk = _kind(tokens[j].group(0))
            if nk is None:
                break
            if nk[0] == "H":
                if cur == 0: break
                cur *= 100
                last = "unit"
            elif nk[0] == "B":
                total += (cur or 1) * nk[1]
                cur = 0
                last = "unit"
            elif nk[0] == "unit":
                if last == "digit": break         # '৩ তিন' -> separate
                if cur and cur % 100 == 0 and nk[1] < 100:
                    cur += nk[1]                  # আঠারশ + বত্রিশ
                elif cur == 0:
                    cur = nk[1]
                else:
                    break
            else:  # digit
                break                             # digits never merge into a span
            j += 1
        value = total + cur
        # drop counter tokens right after the number (সাতটি already handled;
        # this catches '৩৯ টি' / '২১৩২০০ জন')
        k2 = j
        while k2 < len(tokens) and tokens[k2].group(0) in COUNTERS:
            k2 += 1
        parts.append(t[pos:tokens[i].start()])
        parts.append(_fmt(value))
        pos = tokens[k2 - 1].end()
        i = k2
    parts.append(t[pos:])
    return "".join(parts)


PUNCT = r'[।,\.\-‐-―\'"“”‘’()!?;:\s]'


def norm_v2(s):
    """canonicalize numbers, then strip punctuation (keeps # delimiters)."""
    return re.sub(PUNCT, "", canon_numbers(s))


if __name__ == "__main__":
    tests = [
        ("আঠারশ' বত্রিশ সালে", "#1832#সালে"),
        ("দুই লক্ষ তেরো হাজার দুই শত জন", "#213200#"),
        ("৩", "#3#"),
        ("তিন ভাগে", "#3#ভাগে"),
        ("১৮৩২ সালে", "#1832#সালে"),
        ("২১৩২০০", "#213200#"),
        ("প্রায় ১ কোটি", "প্রায়#10000000#"),
        ("১৯৯৪-১৯৯৫ সালের", "#1994##1995#সালের"),
        ("৫,২২,০০০ রুপি", "#522000#রুপি"),
        ("সাতটি", "#7#"),
        ("একটি আখড়া", "#1#আখড়া"),
    ]
    for inp, want in tests:
        got = norm_v2(inp)
        print(("OK " if got == want else "FAIL"), repr(inp), "->", repr(got),
              ("" if got == want else f"  (want {want!r})"))
