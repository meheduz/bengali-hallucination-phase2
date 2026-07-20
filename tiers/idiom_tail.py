"""Closed-book idiom/word-gloss tail tier.

The residual closed-book rows are dominated (114/172) by the template
    "<X>" এর ভাবার্থ কী?      (figurative meaning)
    "<X>" এর শাব্দিক অর্থ কী?  (literal meaning)
and the shipped cb_idiom tier only reaches them when X is one of the 923
entries in assets/bengali_idioms.json — 13 of the 114.  The 32B judge that
currently handles the rest scores 0.714 on the labeled rows of this family.

Reading all 21 labeled rows of the family shows how the hallucinations were
generated, and it is highly structured:

  GLOSS DISPLACEMENT (the dominant mechanism).  A hallucinated response is the
  gloss of a *different* idiom, pasted onto this one:
      অক্কা পাওয়া (= to die)          -> "কর্মব্যস্ততার ভান করা"  (= ব্যস্তবাগীশ)
      সাগর (= sea)                    -> "ঘড়ি ধরে, ঠিক নির্দিষ্ট সময়ে" (= কাঁটায় কাঁটায়)
      নিজের ঢাক নিজে পেটা (= to boast) -> "উন্মুক্তস্থানে শোয়া"
  So the test is not "does the response match this gloss" (which fails on
  legitimate paraphrase: যমে ধরা -> "মারা যাওয়া" vs gloss "মৃত্যুর কবলে পড়া")
  but "does the response match some OTHER dictionary entry markedly better
  than it matches this one".  That is decidable even when X itself is absent
  from the dictionary, which is what lets this reach the 101 residual rows the
  dictionary does not contain.

  LITERAL REUSE.  For শাব্দিক অর্থ (literal) the faithful answers decompose the
  phrase and reuse its own morphemes:
      নিপাতনে সিদ্ধ -> "নিপাতনের মাধ্যমে সিদ্ধি লাভ"        (label 1)
      খটখটে রোদ     -> "খটখট শব্দের মতো তীব্র বা প্রখর রোদ"  (label 1)
  while the hallucinated ones supply an unrelated figurative gloss that reuses
  nothing:
      চরণদাস -> "ব্যঙ্গার্থে তোষামোদকারী"   (label 0)
      টিকা   -> "থাকা; অবস্থান করা"          (label 0)

Sentence embeddings are used only for short gloss-vs-gloss comparison, which
is what they are good at — not for retrieving long questions, where the same
encoder proved to have no discriminative power on this corpus.
"""
import os
import json, re, sys, os, collections, unicodedata, argparse
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
SC = os.environ.get("WORKDIR", ".")
sys.path.insert(0, BASE)
from common import load_test, load_samples

IDIOM_Q = re.compile(r'^[\"“”\'‘’]?(.+?)[\"“”\'‘’]?\s*(?:এ|\')?র\s*(ভাবার্থ|শাব্দিক অর্থ|আভিধানিক অর্থ)\s*(?:কী|কি)\s*\??\s*$')
BN = re.compile(r'[ঀ-৿]+')


def extract(prompt):
    m = IDIOM_Q.match(str(prompt).strip())
    if not m:
        return None, None
    x = m.group(1).strip().strip('"“”\'‘’ ')
    x = re.sub(r'\s*\([^)]*\)\s*$', '', x).strip()   # trailing clarifier
    return (x or None), m.group(2)


# ------------------------------------------------------------------ morphology
# Bengali inflectional endings; stripping them lets 'নিপাতনে' match 'নিপাতনের'
# and 'সিদ্ধ' match 'সিদ্ধি'.  Purely surface morphology, no semantics.
SUFFIX = ['েরই', 'তেই', 'কেই', 'রই', 'ের', 'েতে', 'তে', 'কে', 'য়ে', 'ইয়ে', 'দের',
          'গুলো', 'গুলি', 'টির', 'টি', 'টা', 'খানা', 'রা', 'ের', 'র', 'ি', 'ী',
          'য়', 'ে', 'া', 'ও']


def stem(w):
    w = unicodedata.normalize('NFC', w)
    for s in sorted(SUFFIX, key=len, reverse=True):
        if len(w) > len(s) + 1 and w.endswith(s):
            return w[:-len(s)]
    return w


def stems(s):
    return {stem(w) for w in BN.findall(str(s)) if len(w) > 1}


def prefix_share(a, b, k=3):
    """Do two stems share a k-char prefix?  Catches খটখটে/খটখট, হাবড়হাটি/হাব."""
    return a[:k] == b[:k] and len(a) >= k and len(b) >= k


def literal_reuse(phrase, resp):
    """Fraction of the phrase's content stems echoed by the response."""
    ps, rs = stems(phrase), stems(resp)
    if not ps:
        return 0.0
    hit = 0
    for p in ps:
        if p in rs or any(prefix_share(p, r) for r in rs):
            hit += 1
    return hit / len(ps)


# ------------------------------------------------------------------ dictionary
def load_dict():
    d = json.load(open(os.path.join(BASE, 'assets/bengali_idioms.json')))['idioms']
    return {k.strip(): v for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reuse', type=float, default=0.50)
    ap.add_argument('--disp', type=float, default=0.12,
                    help='margin by which another gloss must beat the queried one')
    ap.add_argument('--selfsim', type=float, default=0.60,
                    help='response-vs-own-gloss similarity that certifies faithful')
    ap.add_argument('--other', type=float, default=0.75,
                    help='absolute similarity another gloss must reach to displace')
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--ship', action='store_true')
    A = ap.parse_args()

    D = load_dict()
    keys = list(D)
    norm_key = {k.replace(' ', ''): k for k in keys}
    print(f'dictionary: {len(D)} entries')

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    cache = SC + '/idiom_gloss_emb.npy'
    if os.path.exists(cache) and len(np.load(cache)) == len(keys):
        G = np.load(cache)
    else:
        G = model.encode([D[k] for k in keys], batch_size=256, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=True)
        np.save(cache, G)
    print('gloss embeddings', G.shape)

    def rows_of(data, enum=False):
        out = []
        for i, r in enumerate(data):
            if r['context']:
                continue
            x, qt = extract(r['prompt_bn'])
            if x:
                out.append((i, r, x, qt))
        return out

    S, T = load_samples(), load_test()
    lab = rows_of(S)
    tst = rows_of(T)
    print(f'labeled family rows: {len(lab)} | test family rows: {len(tst)}')

    def predict_all(rows, REUSE, DISP, SELF, OTHER):
        R = model.encode([r['response_bn'] for _, r, _, _ in rows], batch_size=256,
                         convert_to_numpy=True, normalize_embeddings=True)
        Sim = R @ G.T
        out = []
        for n, (i, r, x, qt) in enumerate(rows):
            own = norm_key.get(x.replace(' ', ''))
            oi = keys.index(own) if own else None
            sims = Sim[n]
            selfsim = float(sims[oi]) if oi is not None else None
            order = np.argsort(-sims)
            # best OTHER entry (exclude the queried one)
            bj = next(int(j) for j in order if j != oi)
            best_other, best_name = float(sims[bj]), keys[bj]

            if qt in ('শাব্দিক অর্থ', 'আভিধানিক অর্থ'):
                ru = literal_reuse(x, r['response_bn'])
                if ru >= REUSE:
                    out.append((1, f'literal-reuse={ru:.2f}', None)); continue
                # no reuse: literal question answered with a foreign gloss
                if best_other >= OTHER:
                    out.append((0, f'literal-displaced->{best_name} {best_other:.2f}', None)); continue
                if selfsim is not None and selfsim >= SELF:
                    # matches its own FIGURATIVE gloss but not the literal form:
                    # answering the wrong sense of the question
                    out.append((0, f'literal-asked-figurative-given {selfsim:.2f}', None)); continue
                out.append((0, f'literal-no-reuse={ru:.2f}', None)); continue

            # ---- ভাবার্থ (figurative)
            if selfsim is not None and selfsim >= SELF and best_other - selfsim < DISP:
                out.append((1, f'gloss-match={selfsim:.2f}', None)); continue
            if selfsim is not None and best_other - selfsim >= DISP and best_other >= OTHER:
                out.append((0, f'displaced->{best_name} {best_other:.2f} vs {selfsim:.2f}', None)); continue
            if selfsim is not None:
                out.append((0 if selfsim < SELF else 1, f'selfsim={selfsim:.2f}', None)); continue
            # not in dictionary: displacement test only
            if best_other >= OTHER:
                out.append((0, f'oov-displaced->{best_name} {best_other:.2f}', None)); continue
            out.append((None, f'oov-nomatch {best_other:.2f}', None))
        return out

    if A.sweep:
        print(f'\n{"reuse":>6}{"disp":>6}{"self":>6}{"other":>6}{"n":>5}{"acc":>8}')
        best = None
        for REUSE in (0.34, 0.50, 0.67):
            for SELF in (0.50, 0.55, 0.60, 0.65, 0.70):
                for OTHER in (0.65, 0.70, 0.75, 0.80):
                    for DISP in (0.05, 0.12, 0.20):
                        pr = predict_all(lab, REUSE, DISP, SELF, OTHER)
                        pairs = [(p, r['label']) for (p, _, _), (_, r, _, _) in zip(pr, lab)
                                 if p is not None]
                        n = len(pairs); ok = sum(1 for p, l in pairs if p == l)
                        a = ok / n if n else 0
                        if best is None or (a, n) > best[0]:
                            best = ((a, n), (REUSE, DISP, SELF, OTHER))
                        print(f'{REUSE:6.2f}{DISP:6.2f}{SELF:6.2f}{OTHER:6.2f}{n:5d}{a:8.4f}')
        print('BEST', best)
        return

    pr = predict_all(lab, A.reuse, A.disp, A.selfsim, A.other)
    pairs = [(p, h, r) for (p, h, _), (_, r, _, _) in zip(pr, lab) if p is not None]
    n = len(pairs); ok = sum(1 for p, h, r in pairs if p == r['label'])
    print(f'\nVALIDATION labeled idiom-family: n={n} acc={ok/n if n else 0:.4f}  '
          f'(32B judge baseline on same rows: 0.7143)')
    for (p, h, _), (i, r, x, qt) in zip(pr, lab):
        mark = ' ' if p == r['label'] else 'X'
        print(f'  {mark} [{i}] {qt[:8]:8s} {x[:22]:22s} pred={p} true={r["label"]}  {h}')

    # split by question type
    for qt in ('ভাবার্থ', 'শাব্দিক অর্থ'):
        sub = [(p, r) for (p, _, _), (_, r, _, q) in zip(pr, lab) if q == qt and p is not None]
        if sub:
            o = sum(1 for p, r in sub if p == r['label'])
            print(f'    {qt}: {o}/{len(sub)} = {o/len(sub):.4f}')

    prt = predict_all(tst, A.reuse, A.disp, A.selfsim, A.other)
    target = set(json.load(open(SC + '/TARGET.json')))
    rows = []
    for (p, h, _), (i, r, x, qt) in zip(prt, tst):
        if p is None:
            continue
        rows.append({'i': i, 'id': int(r['id']), 'pred': p, 'phrase': x, 'qtype': qt,
                     'how': h, 'resid': i in target,
                     'confidence': 'high' if h.startswith(('literal-reuse', 'gloss-match',
                                                           'displaced', 'literal-displaced'))
                                   else 'medium'})
    nres = sum(1 for x in rows if x['resid'])
    print(f'\nTEST family rows covered: {len(rows)}/{len(tst)} | in residual: {nres}')
    print('  pred split:', collections.Counter(x['pred'] for x in rows))
    print('  how split:', collections.Counter(x['how'].split('=')[0].split('->')[0]
                                              for x in rows).most_common())
    json.dump(rows, open(SC + '/idiom_tail_preds.json', 'w'), ensure_ascii=False, indent=1)
    if A.ship:
        json.dump({'meta': {
            'family': 'closed-book idiom/word-gloss template rows',
            'method': 'gloss-displacement test (response matches another dictionary '
                      'entry markedly better than the queried one) + literal-reuse test '
                      'for শাব্দিক অর্থ questions',
            'dictionary': 'work/assets/bengali_idioms.json (923 entries, standard BCS/school canon)',
            'encoder': 'paraphrase-multilingual-MiniLM-L12-v2, used only for short '
                       'gloss-vs-gloss similarity',
            'params': {'reuse': A.reuse, 'disp': A.disp, 'selfsim': A.selfsim, 'other': A.other},
            'validated_n': n, 'validated_acc': round(ok / n, 4) if n else None,
            'baseline_32B_on_same_rows': 0.7143,
        }, 'rows': rows}, open(BASE + '/source_match_cb_tail.json', 'w'),
            ensure_ascii=False, indent=1)
        print('SHIPPED source_match_cb_tail.json')


if __name__ == '__main__':
    main()
