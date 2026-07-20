"""Closed-book idiom/word-gloss tail tier, FINAL (v4) — lexical, no encoder.

Route of arrival, because the negative results matter as much as the tier:

  v1  absolute cosine to the phrase's own gloss              LOO 0.81, n=21
  v2  same, on a dictionary expanded from bn.wiktionary      LOO 0.90, n=21
  v3  rank of the own gloss among all glosses (scale-free)   LOO 0.95, n=21

v3's 0.95 did not survive inspection of its *test* predictions.  Probing the
encoder (paraphrase-multilingual-MiniLM-L12-v2) on this corpus shows it has no
usable Bengali lexical semantics:

(Test-set response text is not reproduced here; each probe row is referred to
by letter.  Glosses shown are public dictionary entries.)

    response A  (a three-synonym gloss whose first synonym is the own gloss)
        vs its OWN gloss "নির্লজ্জ"                        cos 0.79
        vs unrelated     "তোষামোদকারী, হুকুম তামিলকারী"   cos 0.94   <- ranked higher
    response B  (a three-synonym gloss whose last synonym is the own gloss)
        vs its OWN gloss "সংযম"                            cos 0.69
        vs unrelated     "উৎকৃষ্ট, খাঁটি"                  cos 0.84   <- ranked higher

Unrelated glosses outrank near-identical ones, so v3's ranks are noise on any
row where the labeled set did not happen to separate extremely.  Its 0.95 is a
20-row artifact.  The encoder is dropped entirely.

WHAT THIS TIER USES.  Bengali gloss agreement is a *lexical* relation: the
canon writes meanings as short synonym lists, and a faithful response restates
one of those synonyms.  So the tier scores stem overlap between the response
and the phrase's public gloss, with prefix matching to absorb inflection and
compounding (নিরেটবোকা ~ বোকা, খটখটে ~ খটখট).  Two rules:

  শাব্দিক অর্থ (literal)  informative reuse of the phrase's own morphemes -> faithful
  otherwise               stem overlap with the public gloss >= THR -> faithful

and abstain when no public gloss for the phrase exists, leaving the row to the
existing judge rather than guessing.

Every quantity here is inspectable: for any prediction you can print which
stems matched which gloss.  That is the property v3 lacked.
"""
import json, re, sys, os, collections, argparse
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
SC = os.environ.get('WORKDIR', '.')   # dev-only: scratch dir
sys.path.insert(0, BASE)
from common import load_test, load_samples
from idiom_tail import extract, stems, literal_reuse
from idiom_tail2 import merged_glosses, build_rows


def soft_match(a, b, k=3):
    """Stem equality, or a shared >=k-char prefix (inflection / compounding)."""
    if a == b:
        return True
    if len(a) >= k and len(b) >= k and (a.startswith(b) or b.startswith(a)):
        return True
    return False


def overlap(resp, glosses):
    """Best agreement between the response and any one gloss.

    Scored both ways round — a response may be terser than the gloss
    ('সংযম' against a longer synonym list containing it) or richer — and the better
    direction is taken, because either direction shows the same meaning.
    """
    rs = stems(resp)
    if not rs:
        return 0.0, None
    best, which = 0.0, None
    for g in glosses:
        gs = stems(g)
        if not gs:
            continue
        hit_g = sum(1 for t in gs if any(soft_match(t, r) for r in rs))
        hit_r = sum(1 for r in rs if any(soft_match(r, t) for t in gs))
        v = max(hit_g / len(gs), hit_r / len(rs))
        if v > best:
            best, which = v, g
    return best, which


def informative_reuse(phrase, resp, REUSE):
    """Reuse of the phrase's morphemes that also adds new material.

    Guards the loophole where the response merely restates the phrase
    ('চাটনি' -> 'চাটনি (a tangy condiment)'), which is not a gloss.
    """
    ru = literal_reuse(phrase, resp)
    extra = stems(resp) - stems(phrase)
    return (ru >= REUSE and len(extra) >= 2), ru


def annotate(rows, GL, REUSE):
    for row in rows:
        k = row['x'].replace(' ', '')
        gs = GL.get(k, [])
        row['glosses'] = gs
        row['ov'], row['which'] = overlap(row['r']['response_bn'], gs) if gs else (None, None)
        row['ok_reuse'], row['reuse'] = informative_reuse(
            row['x'], row['r']['response_bn'], REUSE)
    return rows


def decide(row, THR):
    if row['qt'] in ('শাব্দিক অর্থ', 'আভিধানিক অর্থ') and row['ok_reuse']:
        return 1, f"literal-reuse={row['reuse']:.2f}"
    if row['ov'] is None:
        return None, 'no-gloss'
    return (1 if row['ov'] >= THR else 0), f"gloss-overlap={row['ov']:.2f}"


def acc_at(rows, THR):
    n = ok = 0
    for row in rows:
        p, _ = decide(row, THR)
        if p is None:
            continue
        n += 1; ok += (p == row['r']['label'])
    return ok, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reuse', type=float, default=0.50)
    ap.add_argument('--thr', type=float, default=0.50)
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--ship', action='store_true')
    A = ap.parse_args()

    GL = merged_glosses()
    print(f'merged public dictionary: {len(GL)} phrases, '
          f'{sum(len(v) for v in GL.values())} glosses')
    S, T = load_samples(), load_test()
    lab = annotate(build_rows(S), GL, A.reuse)
    tst = annotate(build_rows(T), GL, A.reuse)
    print(f'labeled family rows: {len(lab)} ({sum(1 for r in lab if r["glosses"])} with gloss)')
    print(f'test family rows:    {len(tst)} ({sum(1 for r in tst if r["glosses"])} with gloss)')

    grid = np.round(np.arange(0.10, 0.96, 0.05), 2)
    if A.sweep:
        print(f'\n{"THR":>6}{"n":>5}{"acc":>9}')
        for t in grid:
            ok, n = acc_at(lab, t)
            print(f'{t:6.2f}{n:5d}{(ok/n if n else 0):9.4f}')
        return

    # leave-one-out on THR
    loo_n = loo_ok = 0; errs = []
    for i in range(len(lab)):
        rest = lab[:i] + lab[i + 1:]
        bt = max(grid, key=lambda t: (lambda o, n: o / n if n else 0)(*acc_at(rest, t)))
        p, how = decide(lab[i], bt)
        if p is None:
            continue
        loo_n += 1; good = (p == lab[i]['r']['label']); loo_ok += good
        if not good:
            errs.append((lab[i], p, how, bt))

    THR = A.thr
    ok, n = acc_at(lab, THR)
    print(f'\nTHR={THR} (plateau midpoint)   in-sample n={n} acc={ok/n:.4f}')
    print(f'LEAVE-ONE-OUT: n={loo_n} acc={loo_ok/loo_n if loo_n else 0:.4f}   <- honest number')
    print(f'   baselines on the same labeled rows: 32B judge 0.7143 | '
          f'shipped cb_idiom dict-overlap 0.8235')
    for row, p, how, bt in errs:
        print(f'   ERR [{row["i"]}] {row["qt"][:8]} {row["x"][:22]} pred={p} '
              f'true={row["r"]["label"]} {how} thr={bt}')

    print('\nper-row:')
    for row in lab:
        p, how = decide(row, THR)
        mark = ' ' if p == row['r']['label'] else ('~' if p is None else 'X')
        print(f'  {mark} [{row["i"]:3d}] {row["qt"][:8]:8s} {row["x"][:20]:20s} '
              f'pred={p} true={row["r"]["label"]} {how}')

    byqt = collections.defaultdict(lambda: [0, 0])
    for row in lab:
        p, _ = decide(row, THR)
        if p is None:
            continue
        byqt[row['qt']][0] += 1
        byqt[row['qt']][1] += (p == row['r']['label'])
    for q, (c, g) in byqt.items():
        print(f'   {q}: {g}/{c}')

    # ---- test predictions
    target = set(json.load(open(SC + '/TARGET.json')))
    rows = []
    for row in tst:
        p, how = decide(row, THR)
        if p is None:
            continue
        margin = abs((row['ov'] if row['ov'] is not None else 1.0) - THR)
        # Partial literal reuse is the weakest path: a response can echo one of
        # the phrase's words while asserting the opposite sense — কথার বাঁধুনি
        # (= eloquence) answered "কথার কুশ্রীতা" (= ugliness of speech) reuses
        # কথা and passes.  Only full reuse is treated as high confidence.
        if how.startswith('literal-reuse'):
            conf = 'high' if row['reuse'] >= 1.0 else 'medium'
        else:
            conf = 'high' if margin >= 0.25 else 'medium'
        rows.append({'i': row['i'], 'id': int(row['r']['id']), 'pred': p,
                     'phrase': row['x'], 'qtype': row['qt'], 'how': how,
                     'matched_gloss': row['which'], 'n_gloss': len(row['glosses']),
                     'resid': row['i'] in target, 'confidence': conf})
    nres = sum(1 for x in rows if x['resid'])
    print(f'\nTEST covered {len(rows)}/{len(tst)} | in residual {nres} | '
          f'abstained {len(tst)-len(rows)}')
    print('   pred split:', dict(collections.Counter(x['pred'] for x in rows)))
    print('   conf split:', dict(collections.Counter(x['confidence'] for x in rows)))

    json.dump(rows, open(SC + '/idiom_tail4_preds.json', 'w'), ensure_ascii=False, indent=1)
    if A.ship:
        meta = {
            'family': 'closed-book idiom / word-gloss template rows '
                      '("<X> এর ভাবার্থ|শাব্দিক অর্থ কী?")',
            'method': 'morphology-aware lexical stem overlap between the response and the '
                      "phrase's public gloss, plus a guarded literal-morpheme-reuse rule "
                      'for শাব্দিক অর্থ questions; abstains when no public gloss exists',
            'no_encoder': 'sentence embeddings were tried (v1-v3) and dropped: '
                          'paraphrase-multilingual-MiniLM ranks unrelated Bengali glosses '
                          'above near-identical ones (0.94 vs 0.79), so embedding-based '
                          'scores are noise on this corpus.',
            'dictionary': ['work/assets/bengali_idioms.json (curated BCS/school canon, 923)',
                           'work/assets/harvested_gloss.json (bn.wiktionary / bn.wikipedia '
                           'public MediaWiki API, per-phrase, by harvest_gloss.py)'],
            'params': {'THR': THR, 'REUSE': A.reuse},
            'validation': {
                'labeled_n': loo_n,
                'leave_one_out_acc': round(loo_ok / loo_n, 4) if loo_n else None,
                'in_sample_acc': round(ok / n, 4) if n else None,
                'baseline_32B_same_rows': 0.7143,
                'baseline_shipped_cb_idiom_same_family': 0.8235,
                'note': 'n=21 is the entire labeled population of this family, so the '
                        'interval is wide; the tier is preferred because the decision '
                        'plateau is broad (THR 0.35-0.70 all give the same accuracy) '
                        'and every prediction is inspectable.',
            },
        }
        json.dump({'meta': meta, 'rows': rows},
                  open(os.path.join(BASE, 'source_match_cb_tail.json'), 'w'),
                  ensure_ascii=False, indent=1)
        print('SHIPPED source_match_cb_tail.json')


if __name__ == '__main__':
    main()
