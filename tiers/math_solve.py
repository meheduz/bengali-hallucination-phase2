"""Deterministic symbolic solver for Bengali arithmetic word problems.

Each solver below recognizes ONE template by an explicit anchor regex, extracts the
quantities by named capture (never by bare positional order), computes the answer in
exact rational arithmetic (fractions.Fraction), and compares it to the number asserted
by the response.  The verdict is then deterministic:

    response_number == computed_answer  ->  faithful (1)
    otherwise                           ->  hallucinated (0)

A solver returns None (ABSTAIN) whenever anything is ambiguous:
  * the anchor does not match, or a required quantity is missing;
  * the prompt is degenerate -- a zero divisor, or a money amount that has clearly been
    truncated in the source CSV (see MONEY_MIN / analysis/math_solver.md);
  * the response carries no parsable number, or several that disagree;
  * more than one template claims the row.

TEXT NORMALIZATION.  856 of the 2815 dataset rows are NOT in Unicode NFC -- they spell
য় as U+09AF+U+09BC rather than U+09DF, ড় / ঢ় likewise.  Every pattern and every haystack
goes through NFC before matching, otherwise anchors silently never fire.

Usage:
    python math_solve.py     # validate on labeled rows, then emit test predictions
"""
import os
import re, sys, json, math, collections, unicodedata
from fractions import Fraction

sys.path.insert(0, os.environ.get("WORKDIR", os.path.dirname(os.path.abspath(__file__))))
from common import load_samples, load_test

B2E = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


# ---------------------------------------------------------------- text / numbers

def U(s):
    return unicodedata.normalize("NFC", str(s))


def norm(s):
    """NFC, Bengali digits -> ASCII, and drop thousands separators (a comma or thin
    space flanked by digits) so '৩,১০০' reads as 3100."""
    t = U(s).translate(B2E)
    return re.sub(r"(?<=\d)[,  ](?=\d)", "", t)


NUMRE = re.compile(r"\d+(?:\.\d+)?")


def SR(pat, text, flags=0):
    return re.search(U(pat), norm(text), flags)


def SF(pat, text, flags=0):
    return re.findall(U(pat), norm(text), flags)


def has(pat, text):
    return bool(SR(pat, text))


def one(pat, text):
    m = SR(pat, text)
    if not m:
        return None
    try:
        return Fraction(m.group(1))
    except Exception:
        return None


def resp_val(a):
    """The single value asserted by the response, or None."""
    t = norm(a).strip()
    m = re.fullmatch(r"\s*(-)?\s*\(?\s*(\d+)\s*/\s*(\d+)\s*\)?\s*%?\s*", t)
    if m:
        return Fraction(int(m.group(2)), int(m.group(3))) * (-1 if m.group(1) else 1)
    v = set(NUMRE.findall(t))
    if len(v) != 1:
        return None
    x = Fraction(v.pop())
    return -x if t.lstrip().startswith("-") else x


def eq(computed, stated):
    """Exact equality; a rounding allowance only where the true answer is non-integer."""
    if computed is None or stated is None:
        return None
    if computed.denominator == 1:
        return 1 if stated == computed else 0
    return 1 if abs(float(stated) - float(computed)) < 0.005 else 0


def V(computed, resp):
    return (eq(computed, resp), computed) if eq(computed, resp) is not None else None


MONEY_MIN = 10   # money amounts below this are truncated source values -> abstain
DAYS = ["শনিবার", "রবিবার", "সোমবার", "মঙ্গলবার", "বুধবার", "বৃহস্পতিবার", "শুক্রবার"]


# ---------------------------------------------------------------- templates
# Each returns (verdict, computed) or None.

def t_work2(q, a):
    """Two workers, alone in A and B days -> together AB/(A+B)."""
    if not has(r"(দুজনে|উভয়ে).{0,20}(একসাথে|একত্রে|যৌথভাবে)", q):
        return None
    m = SR(r"একা.{0,30}?(\d+)\s*দিনে.{0,60}?একা(?:ই)?.{0,30}?(\d+)\s*দিনে", q, re.S)
    if not m:
        return None
    A, B = Fraction(m.group(1)), Fraction(m.group(2))
    if A == 0 or B == 0:
        return None
    return V(A * B / (A + B), resp_val(a))


def t_work3(q, a):
    """Three workers alone in A, B, C days -> together 1/(1/A+1/B+1/C)."""
    if not has(r"তিনজনে.{0,20}একত্রে", q):
        return None
    m = SR(r"যথাক্রমে\s*(\d+)\s*,\s*(\d+)\s*ও\s*(\d+)\s*দিনে", q)
    if not m:
        return None
    v = [Fraction(m.group(i)) for i in (1, 2, 3)]
    if any(x == 0 for x in v):
        return None
    return V(1 / sum(1 / x for x in v), resp_val(a))


def t_manpower(q, a):
    """M workers finish in D days; E extra workers join -> M*D/(M+E) days."""
    m = SR(r"(\d+)\s*জন লোক\s*(\d+)\s*দিনে", q)
    e = one(r"অতিরিক্ত\s*(\d+)\s*জন", q)
    if not m or e is None:
        return None
    M, D = Fraction(m.group(1)), Fraction(m.group(2))
    if M + e == 0:
        return None
    return V(M * D / (M + e), resp_val(a))


def t_ratio_part(q, a):
    """Ratio a:b over a total, one named part asked.  Covers the age-ratio,
    animal-count and sugar/water-mixture families -- one computation, and which part is
    asked is read off the question word rather than assumed."""
    m = SR(r"অনুপাত\s*(\d+)\s*:\s*(\d+)", q)
    if not m:
        return None
    tot = one(r"(?:সমষ্টি|মোট (?:পশুর সংখ্যা|মাছের সংখ্যা|মিশ্রণ))\s*(\d+)", q)
    if tot is None:
        return None
    A, B = Fraction(m.group(1)), Fraction(m.group(2))
    if A + B == 0:
        return None
    if has(r"(মেয়ের বয়স|ছোট ভাইয়ের বয়স|ছাগলের সংখ্যা|কাতলা মাছের সংখ্যা|পানি কত)", q):
        c = tot * B / (A + B)
    elif has(r"(মায়ের বয়স|বড় ভাইয়ের বয়স|গরুর সংখ্যা|রুই মাছের সংখ্যা|চিনি কত)", q):
        c = tot * A / (A + B)
    else:
        return None
    return V(c, resp_val(a))


def t_partner3(q, a):
    """T taka split among three partners as a:b:c -> the second partner's share."""
    if not has(r"তিন ব্যবসায়িক অংশীদার", q) or not has(r"দ্বিতীয় অংশীদার", q):
        return None
    m = SR(r"(\d+)\s*টাকা\s*তিন", q)
    r = SR(r"(\d+)\s*:\s*(\d+)\s*:\s*(\d+)", q)
    if not (m and r):
        return None
    T = Fraction(m.group(1))
    a1, a2, a3 = (Fraction(r.group(i)) for i in (1, 2, 3))
    if a1 + a2 + a3 == 0:
        return None
    return V(T * a2 / (a1 + a2 + a3), resp_val(a))


def t_sp(q, a):
    """Cost price P sold at r% profit / loss -> selling price."""
    if not has(r"বিক্রয়মূল্য কত", q):
        return None
    P = (one(r"ক্রয়মূল্য\s*(\d+)", q) or one(r"(\d+)\s*টাকায় কেনা", q))
    r = one(r"(\d+)\s*%", q)
    if P is None or r is None or P < MONEY_MIN:
        return None
    if has(r"ক্ষতিতে", q):
        c = P * (1 - r / 100)
    elif has(r"লাভে", q):
        c = P * (1 + r / 100)
    else:
        return None
    return V(c, resp_val(a))


def t_succ_pct(q, a):
    """Price P, then +x%, then -y% applied to the new price."""
    up = one(r"(\d+)\s*%\s*(?:বেড়ে|বৃদ্ধি)", q)
    dn = one(r"(\d+)\s*%\s*(?:কমে|ছাড়)", q)
    P = one(r"(?:শুরুর দাম|প্রাথমিক মূল্য)\s*(\d+)", q)
    if None in (up, dn, P) or P < MONEY_MIN:
        return None
    return V(P * (1 + up / 100) * (1 - dn / 100), resp_val(a))


def t_simple_interest(q, a):
    """Principal P, r% a year, t years -> total simple interest P*r*t/100."""
    if not has(r"সরল সুদ", q) or not has(r"মোট সুদ কত|মোট কত টাকা সুদ", q):
        return None
    P, r, t = one(r"(\d+)\s*টাকা", q), one(r"(\d+)\s*%", q), one(r"(\d+)\s*বছর", q)
    if None in (P, r, t) or P < MONEY_MIN:
        return None
    return V(P * r * t / 100, resp_val(a))


def t_si_multiple(q, a):
    """What simple-interest rate makes any principal n-fold in t years?
    interest (n-1)P = P*r*t/100  ->  r = 100(n-1)/t."""
    if not has(r"সরল সুদের হার", q) or not has(r"সুদে[- ]?আসলে", q):
        return None
    t = one(r"(\d+)\s*বছরে", q)
    n = next((v for k, v in {"দ্বিগুণ": 2, "তিনগুণ": 3, "চারগুণ": 4, "পাঁচগুণ": 5}.items()
              if has(k, q)), None)
    if t is None or n is None or t == 0:
        return None
    return V(Fraction(100 * (n - 1)) / t, resp_val(a))


def t_compound(q, a):
    """P at r% compound for t years -> amount P(1+r/100)^t."""
    if not has(r"চক্রবৃদ্ধি (?:মূলধন|মূল্য)", q):
        return None
    P, r, t = one(r"(\d+)\s*টাকার", q), one(r"(\d+)\s*%", q), one(r"(\d+)\s*বছরের", q)
    if None in (P, r, t) or P < MONEY_MIN:
        return None
    return V(P * (1 + r / 100) ** int(t), resp_val(a))


def t_loss_pct(q, a):
    """Sold for S at a loss of L taka -> loss% = 100L/(S+L)  (profit: 100L/(S-L))."""
    S = one(r"(\d+)\s*টাকায় বিক্রয় কর", q)
    L = one(r"(\d+)\s*টাকা\s*(?:ক্ষতি|লাভ)", q)
    if None in (S, L):
        return None
    if has(r"ক্ষতির শতকরা হার", q):
        c = 100 * L / (S + L)
    elif has(r"লাভের শতকরা হার", q) and S != L:
        c = 100 * L / (S - L)
    else:
        return None
    return V(c, resp_val(a))


SMALL = {"এক": 1, "দুই": 2, "তিন": 3, "চার": 4, "পাঁচ": 5, "ছয়": 6, "সাত": 7,
         "আট": 8, "নয়": 9, "দশ": 10}


def _count(tok):
    """A count written either as digits or as a Bengali numeral word."""
    tok = U(tok).strip()
    if re.fullmatch(r"\d+", tok):
        return Fraction(tok)
    for w, v in SMALL.items():
        if tok in (w, w + "টি", w + "টা"):
            return Fraction(v)
    return None


def t_buy_sell_rate(q, a):
    """Buy n per taka, sell m per taka -> profit% = 100(n-m)/m."""
    if not has(r"শতকরা কত (?:লাভ|ক্ষতি)", q):
        return None
    m = SR(r"টাকায়\s*(\S+?)\s*(?:টি|টা)?\s*করে.{0,40}?ক্রয়.{0,40}?টাকায়\s*(\S+?)\s*(?:টি|টা)?\s*করে.{0,20}?বিক্রয়", q, re.S)
    if not m:
        return None
    n, k = _count(m.group(1)), _count(m.group(2))
    if n is None or k is None or n == 0 or k == 0:
        return None
    cp, sp = 1 / n, 1 / k                       # per-unit cost and selling price
    return V(100 * (sp - cp) / cp, resp_val(a))


def t_rel_speed(q, a):
    """Same direction, speeds v1 and v2, after t hours -> |v1-v2|*t."""
    if not has(r"একই দিকে", q):
        return None
    v = SF(r"(\d+)\s*কিমি", q)
    t = one(r"(\d+)\s*ঘণ্টা পর", q)
    if len(v) != 2 or t is None:
        return None
    return V(abs(Fraction(v[0]) - Fraction(v[1])) * t, resp_val(a))


def t_approach(q, a):
    """Distance D, closing at v1 and v2 -> D/(v1+v2) hours."""
    if not has(r"একে অপরের দিকে", q):
        return None
    D = one(r"দূরত্ব\s*(\d+)\s*কিলোমিটার", q)
    v = SF(r"(\d+)\s*কিমি", q)
    if D is None or len(v) != 2:
        return None
    s = Fraction(v[0]) + Fraction(v[1])
    if s == 0:
        return None
    return V(D / s, resp_val(a))


def t_rate_scale(q, a):
    """Covers d1 feet in t1 seconds -> at the same rate, d2 in t2 = d1*t2/t1."""
    m = SR(r"(\d+)\s*/\s*(\d+)\s*সেকেন্ডে চলে\s*(\d+)\s*ফুট", q)
    t2 = one(r"(\d+)\s*সেকেন্ডে কত ফুট", q)
    if not m or t2 is None:
        return None
    t1 = Fraction(int(m.group(1)), int(m.group(2)))
    d1 = Fraction(m.group(3))
    if t1 == 0:
        return None
    return V(d1 * t2 / t1, resp_val(a))


def t_lcm3(q, a):
    """Three periodic events every A, B, C minutes -> lcm(A,B,C)."""
    if not has(r"(পুনরায় একই মুহূর্তে|আবার একই সময়ে)", q):
        return None
    m = SR(r"প্রতি\s*(\d+)\s*,\s*(\d+)\s*ও\s*(\d+)\s*মিনিট", q)
    if not m:
        return None
    v = [int(m.group(i)) for i in (1, 2, 3)]
    if any(x == 0 for x in v):
        return None
    return V(Fraction(math.lcm(*v)), resp_val(a))


def t_avg_add(q, a):
    """n items averaging m1; one more makes it m2 -> new item = (n+1)m2 - n*m1."""
    n = one(r"(\d+)\s*(?:টি রাশির গড়মান|জন শিক্ষার্থীর গড় নম্বর)", q)
    m1 = one(r"(?:গড়মান|গড় নম্বর)\s*(\d+)", q)
    m2 = one(r"গড়\s*(?:দাঁড়ায়|নম্বর হয়)\s*(\d+)", q)
    if None in (n, m1, m2):
        return None
    return V((n + 1) * m2 - n * m1, resp_val(a))


def t_ncr(q, a):
    """Choose r from n (panel / sub-committee) -> C(n, r)."""
    if not has(r"(প্যানেল গঠন|উপকমিটি গঠন)", q):
        return None
    m = SR(r"(\d+)\s*জন.{0,40}?(\d+)\s*জন", q, re.S)
    if not m:
        return None
    n, r = int(m.group(1)), int(m.group(2))
    if not (0 <= r <= n <= 60):
        return None
    return V(Fraction(math.comb(n, r)), resp_val(a))


def t_dayofweek(q, a):
    """Start day D, k days later -> DAYS[(idx+k) mod 7].  Non-numeric answer."""
    if not has(r"সপ্তাহের কোন (দিন|বার)", q):
        return None
    start = [d for d in DAYS if has(d, q)]
    k = one(r"(\d+)\s*দিন", q)
    if len(start) != 1 or k is None:
        return None
    want = DAYS[(DAYS.index(start[0]) + int(k)) % 7]
    said = [d for d in DAYS if has(d, a)]
    if len(said) != 1:
        return None
    return (1 if said[0] == want else 0), want


def t_age_chain(q, a):
    """Man is d years older than his wife; wife's age = k * son's; in y years the son
    will be s -> son now s-y, wife k(s-y), man k(s-y)+d."""
    d = one(r"স্ত্রীর চেয়ে\s*(\d+)\s*বছরের বড়", q)
    k = one(r"ছেলের বয়সের\s*(\d+)\s*গুণ", q)
    m = SR(r"(\d+)\s*বছর পরে ছেলের বয়স\s*(\d+)\s*বছর", q)
    if None in (d, k) or not m or not has(r"ব্যক্তির বয়স কত", q):
        return None
    y, s = Fraction(m.group(1)), Fraction(m.group(2))
    return V(k * (s - y) + d, resp_val(a))


def t_two_digit(q, a):
    """Two-digit number: units = tens + p, and the number = q*(digit sum) + r."""
    p = one(r"এককের অঙ্ক দশকের অঙ্ক অপেক্ষা\s*(\d+)\s*বেশি", q)
    m = SR(r"অঙ্কদ্বয়ের সমষ্টির\s*(\S+)গুণ অপেক্ষা\s*(\d+)\s*বেশি", q)
    if p is None or not m:
        return None
    mult = {"তিন": 3, "দুই": 2, "চার": 4, "পাঁচ": 5, "ছয়": 6}.get(m.group(1))
    if mult is None:
        return None
    r = Fraction(m.group(2))
    sols = [10 * t + (t + int(p)) for t in range(1, 10)
            if t + int(p) <= 9 and 10 * t + (t + int(p)) == mult * (2 * t + int(p)) + r]
    if len(sols) != 1:
        return None
    return V(Fraction(sols[0]), resp_val(a))


def t_prob_range(q, a):
    """Pick one integer from [lo, hi]: P(prime or multiple of k)."""
    m = SR(r"(\d+)\s*থেকে\s*(\d+)\s*পর্যন্ত সংখ্যা", q)
    k = one(r"(\d+)\s*(?:এর|-এর)\s*গুণিতক", q)
    if not m or k is None or not has(r"সম্ভাবনা", q) or not has(r"মৌলিক অথবা", q):
        return None
    lo, hi = int(m.group(1)), int(m.group(2))
    if not (0 <= lo <= hi <= 10000) or k == 0:
        return None

    def isp(n):
        return n >= 2 and all(n % d for d in range(2, int(n ** 0.5) + 1))

    pool = list(range(lo, hi + 1))
    good = [n for n in pool if isp(n) or n % int(k) == 0]
    return V(Fraction(len(good), len(pool)), resp_val(a))


def t_prime_pick(q, a):
    """'Which of these numbers is prime?' -> the unique prime among the options."""
    if not has(r"কোন সংখ্যাটি মৌলিক", q):
        return None
    opts = [int(x) for x in SF(r"\d+", q)]
    if len(opts) < 2:
        return None

    def isp(n):
        return n >= 2 and all(n % d for d in range(2, int(n ** 0.5) + 1))

    pr = [o for o in opts if isp(o)]
    if len(pr) != 1:
        return None
    return (1 if resp_val(a) == pr[0] else 0), Fraction(pr[0])


def t_frac_of(q, a):
    """p/q of some number equals V -> the number is V*q/p."""
    m = SR(r"(\d+)\s*/\s*(\d+)\s*অংশ\s*(\d+)", q)
    if not m or not has(r"কোন সংখ্যার", q):
        return None
    p, qq, val = (Fraction(m.group(i)) for i in (1, 2, 3))
    if p == 0:
        return None
    return V(val * qq / p, resp_val(a))


def t_two_int_ratio(q, a):
    """Two integers each greater than L sum to S; is the stated ratio admissible?
    The ratio is not unique, so the decidable question is whether the response's ratio
    yields two INTEGERS, both > L, summing to S."""
    m = SR(r"(\d+)\s*(?:হতে|থেকে)\s*বড় দুইটি পূর্ণসংখ্যার যোগফল\s*(\d+)", q)
    if not m or not has(r"অনুপাত কত", q):
        return None
    L, S = int(m.group(1)), int(m.group(2))
    r = SR(r"(\d+)(?:\.\d+)?\s*:\s*(\d+)(?:\.\d+)?", a)
    if not r:
        return None
    x, y = int(r.group(1)), int(r.group(2))
    if x + y == 0:
        return None
    p1, p2 = Fraction(S * x, x + y), Fraction(S * y, x + y)
    ok = (p1.denominator == 1 and p2.denominator == 1 and p1 > L and p2 > L)
    return (1 if ok else 0), f"{p1}+{p2}"


def t_linear_x(q, a):
    """Pure linear single-variable equation, e.g. '5x + 8.5x + 16.5x = 1' -> x."""
    t = norm(q)
    if "**" in t or "^" in t or "²" in t or "³" in t:
        return None
    if not has(r"x এর মান কত", q):
        return None
    m = re.search(r"([0-9x\s\.\+\-\*]+?)=\s*(-?\d+(?:\.\d+)?)", t)
    if not m:
        return None
    lhs, rhs = m.group(1), Fraction(m.group(2))
    terms = re.findall(r"([+-]?)\s*(\d*\.?\d*)\s*\*?\s*x", lhs)
    # the LHS must be exactly the x-terms, nothing else
    if not terms or re.sub(r"[+-]?\s*\d*\.?\d*\s*\*?\s*x", "", lhs).strip():
        return None
    coef = sum(Fraction(-1 if s == "-" else 1) * Fraction(c if c not in ("", ".") else 1)
               for s, c in terms)
    if coef == 0:
        return None
    return V(rhs / coef, resp_val(a))


SOLVERS = [
    t_work2, t_work3, t_manpower, t_ratio_part, t_partner3, t_sp, t_succ_pct,
    t_simple_interest, t_si_multiple, t_compound, t_loss_pct, t_buy_sell_rate,
    t_rel_speed, t_approach, t_rate_scale, t_lcm3, t_avg_add, t_ncr, t_dayofweek,
    t_age_chain, t_two_digit, t_prob_range, t_prime_pick, t_frac_of, t_two_int_ratio,
    t_linear_x,
]


def solve(q, a):
    """-> (verdict, template, computed) or (None, None, None)."""
    hits = []
    for f in SOLVERS:
        try:
            r = f(q, a)
        except Exception:
            r = None
        if r is not None and r[0] is not None:
            hits.append((r[0], f.__name__, r[1]))
    if len(hits) != 1:
        return None, None, None      # abstain on no match or on template collision
    return hits[0]


def fmt(x):
    if isinstance(x, Fraction):
        return int(x) if x.denominator == 1 else round(float(x), 6)
    return x


if __name__ == "__main__":
    S, T = load_samples(), load_test()
    from rules import is_math_prompt

    print("VALIDATION on labeled closed-book rows (dataset samples.json)")
    ok = n = 0
    for i, r in enumerate(S):
        if r["context"]:
            continue
        v, tpl, c = solve(r["prompt_bn"], r["response_bn"])
        if v is None:
            continue
        n += 1
        good = v == r["label"]
        ok += good
        print(f"  {'OK ' if good else 'ERR'} row={i:3d} {tpl:18s} computed={fmt(c)} "
              f"pred={v} true={r['label']} | {r['prompt_bn'][:58]} -> {r['response_bn'][:26]}")
    print(f"\nsolver on labeled rows: {ok}/{n}" + (f" = {ok/n:.3f}" if n else ""))

    labmath = [(i, r) for i, r in enumerate(S)
               if not r["context"] and is_math_prompt(r["prompt_bn"])]
    hit = [i for i, r in labmath if solve(r["prompt_bn"], r["response_bn"])[0] is not None]
    print(f"covers {len(hit)}/{len(labmath)} of the is_math_prompt labeled rows; "
          f"abstains on {[i for i,_ in labmath if i not in hit]}")

    out, tally = [], collections.Counter()
    for i, r in enumerate(T):
        if r["context"]:
            continue
        v, tpl, c = solve(r["prompt_bn"], r["response_bn"])
        if v is None:
            continue
        tally[tpl] += 1
        out.append({"i": i, "pred": v, "template": tpl, "computed": fmt(c),
                    "response_num": fmt(resp_val(r["response_bn"]))})
    json.dump(out, open("source_match_cb_mathsolve.json", "w"), ensure_ascii=False, indent=1)
    print(f"\ntest rows covered: {len(out)}  "
          f"(halluc {sum(1 for o in out if o['pred']==0)} / faithful {sum(1 for o in out if o['pred']==1)})")
    for k, v in tally.most_common():
        print(f"  {k:18s} {v}")

    router = json.load(open("math_test.json"))
    rp = dict(zip(router["idx"], router["pred"]))
    both = [o for o in out if o["i"] in rp]
    flips = [o for o in both if o["pred"] != rp[o["i"]]]
    print(f"\nvs thinking-mode router: overlap {len(both)}, flips {len(flips)} -> "
          f"{[(o['i'], rp[o['i']], o['pred']) for o in flips]}")
    ex = json.load(open("source_match_cb_math_extra_UNVALIDATED.json"))
    ep = {d["i"]: d["pred"] for d in ex}
    b2 = [o for o in out if o["i"] in ep]
    f2 = [o for o in b2 if o["pred"] != ep[o["i"]]]
    print(f"vs math_extra: overlap {len(b2)}, flips {len(f2)} -> "
          f"{[(o['i'], ep[o['i']], o['pred']) for o in f2]}")
