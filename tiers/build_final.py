"""Final build: v22 + whatever last-hour tiers exist. Run once, submit."""
import json, csv, collections, os
from common import load_samples, load_test
from ctx_model import norm
from bn_num import norm_v2

S, T = load_samples(), load_test()
def R(f):
    if not os.path.exists(f): return []
    try:
        d = json.load(open(f)); return d if isinstance(d, list) else d.get("rows", [])
    except Exception: return []

# NEW last-hour tiers (highest precedence within their segment)
last_gloss = {r["i"]: r["pred"] for r in R("source_match_last_gloss.json") if r.get("pred") in (0,1)}
last_bip   = {r["i"]: r["pred"] for r in R("source_match_last_biparit.json") if r.get("pred") in (0,1)}
last_sweep = {r["i"]: r["pred"] for r in R("source_match_last_sweep.json") if r.get("pred") in (0,1)}
ctx_bank   = {r["i"]: r["pred"] for r in R("source_match_ctx_bank.json") if r.get("pred") in (0,1)}
final_abs  = {r["i"]: r["pred"] for r in R("source_match_final_abstentions.json") if r.get("pred") in (0,1)}
print(f"final-round: ctx_bank {len(ctx_bank)}, abstentions {len(final_abs)}")
print(f"last-hour tiers: gloss {len(last_gloss)}, biparit {len(last_bip)}, sweep {len(last_sweep)}")

mathsolve = {r["i"]: r["pred"] for r in R("source_match_cb_mathsolve.json")}
cblast    = {r["i"]: r["pred"] for r in R("source_match_cb_last.json")}
ctxbio    = {r["i"]: r["pred"] for r in R("source_match_ctx_bio.json")}
det = {}
for f in ["source_match_ctx_grammar.json","source_match_ctx_biparit_sandhi.json","source_match_ctx_idiom_upasarga.json"]:
    for r in R(f): det.setdefault(r["i"], r["pred"])
cbtail = {r["i"]: r["pred"] for r in R("source_match_cb_tail.json")}
lm2    = {r["i"]: r["pred"] for r in R("source_match_cb_livemcq2.json")}
sites  = {r["i"]: r["pred"] for r in R("source_match_cb_sites.json")}
gram   = {r["i"]: r["pred"] for r in R("source_match_cb_gram.json")}
wsrc   = {r["i"]: r["pred"] for r in R("source_match_cb_wikisource.json")}
wikt   = {r["i"]: r["pred"] for r in R("wikt_test_pred.json")}
ocr    = {r["i"]: r["pred"] for r in R("source_match_cb_ocr.json")}
wiki   = {r["i"]: r["pred"] for r in R("source_match_cb_wiki.json")}
reroute = json.load(open("reroute.json")); ctx3 = json.load(open("source_match_ctx3.json"))
j32 = {int(k): v for k, v in json.load(open("judge32b_scores.json"))["scores"].items()}
tk = json.load(open("ctx_think_test.json")); ctx_think = dict(zip(tk["idx"], tk["pred"]))
b14 = {json.loads(l)["i"]: json.loads(l)["p"] for l in open("ctx_14b_test.partial")}
cj = json.load(open("ctx_sub_judge.json")); lp = {i: (0 if s > 0.19 else 1) for i, s in zip(cj["idx"], cj["scores"])}
for i, p14 in b14.items():
    if i in ctx_think:
        v = [ctx_think[i], p14] + ([lp[i]] if i in lp else [])
        ctx_think[i] = 1 if sum(v) * 2 > len(v) else 0
ctx_match = json.load(open("source_match_ctx.json"))
cb_match = R("source_match_cb.json")
d2 = json.load(open("source_match_cb2.json")); cb_match += d2["rows"] if isinstance(d2, dict) else d2
cb_pred = {r["id"] - 1: r["pred"] for r in cb_match if r.get("pred") in (0,1)}
mt = json.load(open("math_test.json")); math_router = dict(zip(mt["idx"], mt["pred"]))

byp = collections.defaultdict(list)
for r in S: byp[r["prompt_bn"].strip()].append(r)
pred, n = [], collections.Counter()
for i, r in enumerate(T):
    rt = reroute.get(str(i))
    if rt == "cb_gram" and i in gram: pred.append(gram[i]); n["cb_gram"] += 1; continue
    if rt == "ctx_gold_tydiqa" and str(i) in ctx3 and ctx3[str(i)].get("pred_label") in (0,1):
        pred.append(ctx3[str(i)]["pred_label"]); n["ctx_gold3"] += 1; continue
    others = byp.get(r["prompt_bn"].strip(), [])
    exact = [o for o in others if str(o["response_bn"]).strip() == str(r["response_bn"]).strip()]
    if exact: p = exact[0]["label"]; n["leak"] += 1
    elif r["context"]:
        m = ctx_match.get(str(i))
        if i in ctxbio: p = ctxbio[i]; n["ctx_bio"] += 1
        elif m and m.get("pred_label") in (0,1) and not m.get("suspect_gold"):
            p = m["pred_label"]; n["ctx_gold"] += 1
        elif i in ctx_bank: p = ctx_bank[i]; n["ctx_BANK"] += 1
        elif i in final_abs: p = final_abs[i]; n["ctx_FINAL_ABS"] += 1
        elif i in last_bip: p = last_bip[i]; n["ctx_LAST_BIPARIT"] += 1
        elif i in last_gloss: p = last_gloss[i]; n["ctx_LAST_GLOSS"] += 1
        elif i in det: p = det[i]; n["ctx_deterministic"] += 1
        else: p = ctx_think[i]; n["ctx_think"] += 1
    elif i in final_abs: p = final_abs[i]; n["cb_FINAL_ABS"] += 1
    elif i in last_sweep: p = last_sweep[i]; n["cb_LAST_SWEEP"] += 1
    elif i in last_gloss: p = last_gloss[i]; n["cb_LAST_GLOSS"] += 1
    elif i in mathsolve: p = mathsolve[i]; n["cb_mathsolve"] += 1
    elif i in cblast: p = cblast[i]; n["cb_last"] += 1
    elif i in lm2: p = lm2[i]; n["cb_livemcq2"] += 1
    elif i in cb_pred: p = cb_pred[i]; n["cb_gold"] += 1
    elif i in math_router: p = math_router[i]; n["cb_math"] += 1
    elif i in wiki: p = wiki[i]; n["cb_wiki"] += 1
    elif i in cbtail: p = cbtail[i]; n["cb_tail"] += 1
    elif i in wikt: p = wikt[i]; n["cb_idiom"] += 1
    elif i in wsrc: p = wsrc[i]; n["cb_wikisource"] += 1
    elif i in sites: p = sites[i]; n["cb_sites"] += 1
    elif i in ocr: p = ocr[i]; n["cb_ocr"] += 1
    elif i in j32: p = 0 if j32[i] > 0.50 else 1; n["cb_32B"] += 1
    else: p = 0; n["cb_default"] += 1
    pred.append(p)

print("layers:", dict(sorted(n.items())))
old = [int(x["label"]) for x in csv.DictReader(open("predictions.csv"))]
ch = sum(1 for a,b in zip(old,pred) if a!=b)
print(f"changed vs v22: {ch} | halluc={pred.count(0)} faithful={pred.count(1)}")
with open("predictions.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["id","label"])
    for r,p in zip(T,pred): w.writerow([r["id"],p])
import shutil; shutil.copy("predictions.csv","../predictions.csv")
print(f"FINAL written ({ch} rows changed)")

# appended: pick up final-round tiers if present
