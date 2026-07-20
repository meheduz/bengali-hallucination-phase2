"""Closed-book idiom/word-gloss tail tier, v2 — merged public dictionary.

v1 (idiom_tail.py) validated 0.8095 on the 21 labeled rows of this family but
routed most test rows down an `oov-displaced` path validated on n=1, because
assets/bengali_idioms.json contains only 28 of the 150 phrases the test asks
about.  harvest_gloss.py fixes the cause rather than the symptom: it pulls the
gloss for each asked-about phrase straight from bn.wiktionary (public API), and
in-dictionary coverage goes 28 -> 101 of 150.  Rows therefore land on the
gloss-similarity path that the labeled rows actually validate.

DECISION RULE (deliberately kept to two parameters, because n=21 labels cannot
support a richer one):

  শাব্দিক অর্থ (literal)   literal_reuse(phrase, response) >= REUSE  -> faithful
                          otherwise fall through to the gloss test
  gloss test              selfsim = max cos(response, g) for g in glosses(X)
                          selfsim >= TAU -> faithful, else hallucinated
  no gloss available      abstain (do NOT guess; leave the row to the 32B judge)

Accuracy is reported leave-one-out: TAU is refit on the other 20 rows for each
held-out row, so the number is not the in-sample optimum.
"""
import os
import json, re, sys, os, collections, argparse
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
SC = os.environ.get("WORKDIR", ".")
sys.path.insert(0, BASE)
from common import load_test, load_samples
from idiom_tail import extract, literal_reuse, load_dict

ENCODER = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'


def merged_glosses():
    """phrase -> list of gloss strings, from the curated canon + bn.wiktionary."""
    D = load_dict()
    out = collections.defaultdict(list)
    for k, v in D.items():
        out[k.replace(' ', '')].append(v)
    H = json.load(open(os.path.join(BASE, 'assets/harvested_gloss.json')))
    for title, rec in H.items():
        k = title.replace(' ', '')
        for g in rec.get('gloss', []):
            if g and len(g) > 1:
                out[k].append(g)
        for g in rec.get('sections', {}).values():
            if g:
                out[k].append(g)
    return {k: list(dict.fromkeys(v)) for k, v in out.items()}


def build_rows(data):
    out = []
    for i, r in enumerate(data):
        if r['context']:
            continue
        x, qt = extract(r['prompt_bn'])
        if x:
            out.append({'i': i, 'r': r, 'x': x, 'qt': qt})
    return out


def score(rows, GL, model):
    """Attach selfsim + literal reuse to every row."""
    texts, owner = [], []
    for n, row in enumerate(rows):
        gs = GL.get(row['x'].replace(' ', ''), [])
        row['glosses'] = gs
        for g in gs:
            texts.append(g); owner.append(n)
    R = model.encode([row['r']['response_bn'] for row in rows], batch_size=128,
                     convert_to_numpy=True, normalize_embeddings=True)
    if texts:
        Gv = model.encode(texts, batch_size=128, convert_to_numpy=True,
                          normalize_embeddings=True)
    best = [None] * len(rows)
    for k, n in enumerate(owner):
        s = float(R[n] @ Gv[k])
        if best[n] is None or s > best[n]:
            best[n] = s
    for n, row in enumerate(rows):
        row['selfsim'] = best[n]
        row['reuse'] = literal_reuse(row['x'], row['r']['response_bn'])
    return rows


def decide(row, TAU, REUSE):
    if row['qt'] in ('শাব্দিক অর্থ', 'আভিধানিক অর্থ') and row['reuse'] >= REUSE:
        return 1, f"literal-reuse={row['reuse']:.2f}"
    if row['selfsim'] is None:
        return None, 'no-gloss'
    return (1 if row['selfsim'] >= TAU else 0), f"selfsim={row['selfsim']:.2f}"


def fit_tau(rows, REUSE, grid):
    best, bt = -1, grid[len(grid) // 2]
    for t in grid:
        n = ok = 0
        for row in rows:
            p, _ = decide(row, t, REUSE)
            if p is None:
                continue
            n += 1; ok += (p == row['r']['label'])
        a = ok / n if n else 0
        if a > best:
            best, bt = a, t
    return bt, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reuse', type=float, default=0.50)
    ap.add_argument('--tau', type=float, default=None)
    ap.add_argument('--ship', action='store_true')
    A = ap.parse_args()

    GL = merged_glosses()
    print(f'merged dictionary: {len(GL)} phrases, '
          f'{sum(len(v) for v in GL.values())} glosses')

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(ENCODER)

    S, T = load_samples(), load_test()
    lab = score(build_rows(S), GL, model)
    tst = score(build_rows(T), GL, model)
    print(f'labeled family rows: {len(lab)} '
          f'({sum(1 for r in lab if r["selfsim"] is not None)} with a gloss)')
    print(f'test family rows:    {len(tst)} '
          f'({sum(1 for r in tst if r["selfsim"] is not None)} with a gloss)')

    grid = np.round(np.arange(0.50, 0.951, 0.005), 3)

    # ---- leave-one-out validation (TAU refit without the held-out row)
    loo_n = loo_ok = 0
    loo_err = []
    for k in range(len(lab)):
        rest = lab[:k] + lab[k + 1:]
        t, _ = fit_tau(rest, A.reuse, grid)
        p, how = decide(lab[k], t, A.reuse)
        if p is None:
            continue
        loo_n += 1
        good = (p == lab[k]['r']['label'])
        loo_ok += good
        if not good:
            loo_err.append((lab[k], p, how, t))

    TAU, insample = fit_tau(lab, A.reuse, grid) if A.tau is None else (A.tau, None)
    print(f'\nfitted TAU={TAU}  (in-sample acc {insample:.4f})' if insample is not None
          else f'\nTAU={TAU} (given)')
    print(f'LEAVE-ONE-OUT: n={loo_n} acc={loo_ok/loo_n if loo_n else 0:.4f}   '
          f'<- the honest number')
    print(f'   32B judge on the same labeled rows: 0.7143')
    for row, p, how, t in loo_err:
        print(f'   ERR [{row["i"]}] {row["qt"][:8]} {row["x"][:24]} pred={p} '
              f'true={row["r"]["label"]} {how} (tau={t})')

    # in-sample breakdown for transparency
    n = ok = 0
    byqt = collections.defaultdict(lambda: [0, 0])
    byhow = collections.defaultdict(lambda: [0, 0])
    for row in lab:
        p, how = decide(row, TAU, A.reuse)
        if p is None:
            continue
        n += 1; good = (p == row['r']['label']); ok += good
        byqt[row['qt']][0] += 1; byqt[row['qt']][1] += good
        h = how.split('=')[0]
        byhow[h][0] += 1; byhow[h][1] += good
    print(f'\nin-sample at TAU={TAU}: n={n} acc={ok/n:.4f}')
    for q, (c, g) in byqt.items():
        print(f'   {q}: {g}/{c}')
    for h, (c, g) in byhow.items():
        print(f'   path {h}: {g}/{c}')

    # ---- test predictions
    target = set(json.load(open(SC + '/TARGET.json')))
    rows = []
    for row in tst:
        p, how = decide(row, TAU, A.reuse)
        if p is None:
            continue
        margin = abs((row['selfsim'] or 0) - TAU)
        rows.append({'i': row['i'], 'id': int(row['r']['id']), 'pred': p,
                     'phrase': row['x'], 'qtype': row['qt'], 'how': how,
                     'n_gloss': len(row['glosses']),
                     'resid': row['i'] in target,
                     'confidence': 'high' if how.startswith('literal-reuse') or margin >= 0.08
                                   else 'medium'})
    nres = sum(1 for x in rows if x['resid'])
    print(f'\nTEST covered {len(rows)}/{len(tst)} | in residual {nres} | '
          f'abstained {len(tst)-len(rows)}')
    print('   pred split:', dict(collections.Counter(x['pred'] for x in rows)))
    print('   conf split:', dict(collections.Counter(x['confidence'] for x in rows)))

    json.dump(rows, open(SC + '/idiom_tail2_preds.json', 'w'), ensure_ascii=False, indent=1)
    if A.ship:
        meta = {
            'family': 'closed-book idiom / word-gloss template rows '
                      '("<X> এর ভাবার্থ|শাব্দিক অর্থ কী?")',
            'method': 'gloss similarity against a merged public dictionary, plus a '
                      'literal-morpheme-reuse test for শাব্দিক অর্থ questions; abstain '
                      'when no public gloss exists for the phrase',
            'dictionary': ['work/assets/bengali_idioms.json (curated BCS/school canon)',
                           'work/assets/harvested_gloss.json (bn.wiktionary / bn.wikipedia '
                           'public MediaWiki API, harvested per-phrase by harvest_gloss.py)'],
            'encoder': ENCODER + ' (short gloss-vs-gloss similarity only)',
            'params': {'TAU': float(TAU), 'REUSE': A.reuse},
            'validation': {
                'labeled_n': loo_n,
                'leave_one_out_acc': round(loo_ok / loo_n, 4) if loo_n else None,
                'in_sample_acc': round(ok / n, 4) if n else None,
                'baseline_32B_same_rows': 0.7143,
                'note': 'leave-one-out refits TAU on the other rows; it is the number '
                        'to trust. n=21 is the entire labeled population of this family.',
            },
        }
        json.dump({'meta': meta, 'rows': rows},
                  open(os.path.join(BASE, 'source_match_cb_tail.json'), 'w'),
                  ensure_ascii=False, indent=1)
        print('SHIPPED source_match_cb_tail.json')


if __name__ == '__main__':
    main()
