# অলীকবচন — Bengali LLM Hallucination Detection

**Team: Ain't No Way — Private LB 0.958, rank 5 of 213** (public 0.954)

Phase-2 submission package: inference notebook, tier modules, paper, and presentation.

## Approach

We did not build a stronger hallucination judge. We established the benchmark's
**provenance** — identifying the public sources its items were drawn from, and verifying
each response against those sources rather than asking a model to guess.

The entry point was a fingerprint the dataset builders left behind: two test rows retain
markdown image URLs pointing at `web.livemcq.com`, a Bangladeshi exam-solution site.
Following it revealed that a large share of the benchmark is recoverable from public data.

### Layer stack (2,516 test rows)

| Layer | Rows | Basis | Validated |
|---|---|---|---|
| Exact leak | 11 | (prompt, response) match in released sample split | — |
| ctx_gold | 922 | `csebuetnlp/squad_bn` + TyDiQA-GoldP-bn gold-answer containment | 0.983 |
| ctx_deterministic | 191 | Bengali grammar canon (সমাস, সন্ধি, বাগধারা, উপসর্গ, বিপরীত) | rule-exact |
| ctx_think | 223 | Qwen3-8B thinking + 14B 3-judge arbitration | 0.875 F1 (n=40) |
| cb_livemcq2 | 711 | livemcq BCS bank (8,214 Q/A, public WordPress API) | 97/97 |
| cb_mathsolve | 139 | 26 symbolic templates, exact `Fraction` arithmetic | 10/10 (n=10) |
| cb_tail | 106 | Bengali idiom-dictionary gloss overlap | 0.950 LOO (n=20) |
| cb_wiki | 62 | bn.wikipedia article containment | 3/3 (n=3) |
| cb_last | 47 | wiki infobox facts, statute clauses, closed-set canon | sourced |
| remaining | 104 | bangla-mmlu, wiktionary, wikisource, sibling exam banks, 32B judge | — |

An LLM decides 236 of 2,516 rows (9.4%) — 223 `ctx_think`, 11 the closed-book
thinking math router, 2 the 32B judge. The other 2,280 (90.6%) are a source
lookup or a deterministic rule.

### Key deterministic rules

- **সমাস**: the context supplies the ব্যাসবাক্য; its joining particle fixes the type
  (ও → দ্বন্দ্ব, genitive → ষষ্ঠী তৎপুরুষ, দ্বারা → তৃতীয়া, হইতে → পঞ্চমী, জন্য → চতুর্থী).
- **বাগধারা**: idiom canon with a polarity guard — কাক নিদ্রা is *light* sleep, so a
  response saying গভীর নিদ্রা is hallucinated regardless of lexical overlap.
- **Arithmetic**: 800 × 8 × 4 / 100 = 256, so a response of ১,৫৩৬ is hallucinated.

## Measured negative results

- Closed-book Bangladesh-specific knowledge caps at ~0.77 across Qwen3-8B/14B,
  Gemma-3-12B, Qwen3-30B-A3B and Qwen3-32B — with thinking-mode **and** retrieval.
  Model scale does not fix it.
- Self-consistency (5-sample majority vote) **hurt**: ctx 0.784 vs 0.875 greedy.
- QLoRA fine-tuning on gold-pseudo-labels collapsed (ctx 0.610, cb 0.216).
- BanglaBERT pseudo-label training failed its gates (0.765 / 0.690).

## Anti-overfitting protocol

Every tier was gated on the released 299-row labeled split before shipping. Proposed and
**rejected** on measured evidence: leak-flip rule (n=4, precision CI lower bound 0.56),
logistic stacks, tuned-LCS threshold, 14B logprob ensemble, embedding-based question
matching (0.855), self-consistency, two fine-tuning approaches. No leaderboard probing —
every threshold was fit on the sample split, never on leaderboard feedback.

Result: the private score (0.958) came in **above** the public score (0.954).

## Integrity note

During the source search we found a publicly listed Kaggle dataset whose own manifest
describes ~37,740 rows as `labeled_test_derived_verified_synthetic`, covering all 418
then-unmatched context rows with labels attached. Using it would have been trivial. We
refused it (competition rules: *"using the test set labels in any form is not allowed"*)
and quarantined our matching output unused. It was never loaded by any tier and never
influenced any submission. The 0.958 was reached without it.

## Repository layout

```
notebook/   phase2_notebook.py    — inference notebook (public-rerun + held-out modes)
            phase2_inference.ipynb — the same notebook in .ipynb form
            NOTEBOOK_README.md    — dataset manifest, runtime story, compliance notes
tiers/      *.py                  — tier modules imported by the notebook
```

## Required Kaggle datasets

The notebook expects these attached (public):

- `mdmeheduzzaman/bengali-halluc-v23-artifacts` — 27 router artifacts + reference CSV
- `mdmeheduzzaman/bengali-halluc-source-banks` — exam banks, idiom canon, grammar KB
- `mdmeheduzzaman/bengali-halluc-tier-code` — tier modules
- `mdmeheduzzaman/bengali-halluc-dataset-samples` — the released 299-row labelled
  split. OPTIONAL: it is absent from the held-out fold, and the notebook detects
  that and runs without it (the exact-leak layer and the validation printouts are
  skipped; no prediction path depends on it).

Model: `qwen-lm/qwen-3/transformers/8b/1` (Qwen3-8B, 4-bit at load). Attached as a
Kaggle model source, so no download is needed and the kernel runs with internet off.
`Qwen/Qwen3-32B` is optional and only affects 2 rows.

Models: `Qwen/Qwen3-8B` (4-bit) required; `Qwen/Qwen3-32B` (4-bit) optional. No model was
fine-tuned — the "checkpoint" is open weights plus deterministic match indices.

## Runtime

Public rerun (2,516 rows, precomputed artifacts): minutes.
Live recompute (2,516 rows): ~3 h on 2×T4.
Held-out rehearsal (5,032 rows): 5.97 h projected at full thinking-mode, inside the 9 h cap,
with a tested three-rung degradation ladder (thinking → logprob → substring rule).

## Data sources cited

csebuetnlp/squad_bn · TyDiQA-GoldP-bn (Clark et al. 2020) · hishab/bangla-mmlu (TituLLMs,
Nahin et al. 2025) · BanglaRQA (Ekram et al. 2022) · bn.wikipedia · bn.wiktionary ·
bn.wikisource (জ্ঞানেন্দ্রমোহন দাস) · accessibledictionary.gov.bd · bdlaws.minlaw.gov.bd ·
web.livemcq.com · jobstestbd.com · onlinebcs.com · kalikolom.com · dailyshikkha.com
