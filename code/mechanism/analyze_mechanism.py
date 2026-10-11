"""Exploratory mechanism analysis of FERT-REFUND run 2 (post hoc; not preregistered).

python analyze_mechanism.py <unzipped run 2 dir with qwen/ gemma/> <out dir>
  -> mechanism.json, MECHANISM_TABLES.md

Uses each valid item's step-1..5 acceptance matrix (record "acc") and token tags ("tags"):
  M1  decomposition of the step-1 acceptance gap to English into token-type composition and
      within-type acceptance (a two-fold Oaxaca split with English weights);
  M2  conditional acceptance by draft depth, P(draft j accepted | drafts 1..j-1 accepted);
  M3  script groups;
  M4  byte-fragment tokens: share and step-1 acceptance.
Anchor a (output position a) drafts y[a+1..a+5]; draft j is scored only where y[a+j] exists.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "probe"))
import fert_lib as fl  # noqa: E402

TAGS = ("word_start", "cont", "frag", "other", "special")
SCRIPT = {"Latin (non-English)": ["fra_Latn", "deu_Latn", "fin_Latn", "tur_Latn", "swh_Latn"],
          "Cyrillic/Arabic": ["rus_Cyrl", "arb_Arab"],
          "Indic": ["hin_Deva", "ben_Beng", "guj_Gujr", "tam_Taml", "tel_Telu"],
          "Ethiopic/Georgian": ["amh_Ethi", "kat_Geor"],
          "SE Asian": ["tha_Thai", "mya_Mymr", "khm_Khmr"],
          "CJK": ["zho_Hans", "jpn_Jpan", "kor_Hang"]}
N_BOOT = 1000


def read_jsonl(p: Path):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def item_counts(r) -> dict:
    """Per-item sufficient statistics: step-1 accepts/totals by tag, depth-conditional counts."""
    acc = fl.decode_acc(r["acc"])
    n, tags = r["n"], r["tags"]
    c = {"s1": {t: [0, 0] for t in TAGS}, "depth": [[0, 0] for _ in range(fl.K_MAX)], "tok": {t: 0 for t in TAGS}}
    for t in tags:
        c["tok"][t] += 1
    for a in range(acc.shape[0]):
        if a + 1 <= n - 1:
            t = tags[a + 1]
            c["s1"][t][0] += int(acc[a, 0])
            c["s1"][t][1] += 1
        for j in range(1, fl.K_MAX + 1):
            if a + j > n - 1:
                break
            if j > 1 and not acc[a, :j - 1].all():
                break
            c["depth"][j - 1][0] += int(acc[a, j - 1])
            c["depth"][j - 1][1] += 1
    return c


def pool(cs) -> dict:
    out = {"s1": {t: [0, 0] for t in TAGS}, "depth": [[0, 0] for _ in range(fl.K_MAX)], "tok": {t: 0 for t in TAGS}}
    for c in cs:
        for t in TAGS:
            out["s1"][t][0] += c["s1"][t][0]
            out["s1"][t][1] += c["s1"][t][1]
            out["tok"][t] += c["tok"][t]
        for j in range(fl.K_MAX):
            out["depth"][j][0] += c["depth"][j][0]
            out["depth"][j][1] += c["depth"][j][1]
    return out


def shares_rates(p):
    tot = sum(p["s1"][t][1] for t in TAGS)
    pi = {t: p["s1"][t][1] / tot for t in TAGS}
    rate = {t: (p["s1"][t][0] / p["s1"][t][1]) if p["s1"][t][1] else 0.0 for t in TAGS}
    acc1 = sum(p["s1"][t][0] for t in TAGS) / tot
    return pi, rate, acc1


def oaxaca(pL, pE) -> dict:
    """acc1_L - acc1_en = composition + within. Composition uses English rates, within uses L shares.
    A tag present in L but absent in English gets L's rate in both terms (composition only via share)."""
    piL, rL, aL = shares_rates(pL)
    piE, rE, aE = shares_rates(pE)
    rE_f = {t: rE[t] if pE["s1"][t][1] else rL[t] for t in TAGS}
    comp = sum((piL[t] - piE[t]) * rE_f[t] for t in TAGS)
    within = sum(piL[t] * (rL[t] - rE_f[t]) for t in TAGS)
    return {"gap": aL - aE, "composition": comp, "within": within, "acc1": aL, "acc1_en": aE,
            "shares": piL, "rates": rL}


def boot_oaxaca(cL, cE, seed=0):
    rng = np.random.default_rng(seed)
    comp, within = [], []
    for _ in range(N_BOOT):
        iL = rng.integers(0, len(cL), len(cL))
        iE = rng.integers(0, len(cE), len(cE))
        o = oaxaca(pool([cL[i] for i in iL]), pool([cE[i] for i in iE]))
        comp.append(o["composition"])
        within.append(o["within"])
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return q(comp), q(within)


def family(fdir: Path, t_ratio: dict) -> dict:
    counts = {}
    for L in fl.LANGS:
        p = fdir / f"{L}.jsonl"
        if p.exists():
            counts[L] = [item_counts(r) for r in read_jsonl(p) if r["valid"]]
    pooled = {L: pool(c) for L, c in counts.items()}
    E = pooled["eng_Latn"]
    res = {"langs": {}}
    for L, p in pooled.items():
        o = oaxaca(p, E)
        if L != "eng_Latn":
            o["composition_ci"], o["within_ci"] = boot_oaxaca(counts[L], counts["eng_Latn"])
        o["cond_depth"] = [d[0] / d[1] if d[1] else float("nan") for d in p["depth"]]
        tok = sum(p["tok"].values())
        o["frag_share"] = p["tok"]["frag"] / tok
        o["frag_acc1"] = o["rates"]["frag"] if p["s1"]["frag"][1] else None
        o["T_ratio"] = t_ratio.get(L)
        o["n_items"] = len(counts[L])
        res["langs"][L] = o
    non_en = [L for L in res["langs"] if L != "eng_Latn"]
    res["mean_gap"] = float(np.mean([res["langs"][L]["gap"] for L in non_en]))
    res["mean_composition"] = float(np.mean([res["langs"][L]["composition"] for L in non_en]))
    res["mean_within"] = float(np.mean([res["langs"][L]["within"] for L in non_en]))
    res["n_within_negative_ci"] = sum(res["langs"][L]["within_ci"][1] < 0 for L in non_en)
    res["n_composition_positive_ci"] = sum(res["langs"][L]["composition_ci"][0] > 0 for L in non_en)
    res["scripts"] = {}
    for g, Ls in SCRIPT.items():
        Ls = [L for L in Ls if L in res["langs"]]
        res["scripts"][g] = {"langs": Ls, "acc1": float(np.mean([res["langs"][L]["acc1"] for L in Ls])),
                             "within": float(np.mean([res["langs"][L]["within"] for L in Ls])),
                             "composition": float(np.mean([res["langs"][L]["composition"] for L in Ls]))}
    return res


def tables(res) -> str:
    out = []
    for f, r in res.items():
        out.append(f"\n## {f}\n")
        out.append(f"Mean over 20 non-English languages: gap {r['mean_gap']:+.3f} = composition "
                   f"{r['mean_composition']:+.3f} + within-type {r['mean_within']:+.3f}. Within-type CI wholly "
                   f"below 0 in {r['n_within_negative_ci']}/20; composition CI wholly above 0 in "
                   f"{r['n_composition_positive_ci']}/20.\n")
        out.append("| lang | T/T_en | acc1 | gap | composition [CI] | within [CI] | cont share | frag share | "
                   "P(d2|d1) | P(d3|d1,2) |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for L, o in sorted(r["langs"].items(), key=lambda kv: kv[1]["T_ratio"] or 0):
            ci = (lambda k: f" [{o[k][0]:+.3f}, {o[k][1]:+.3f}]" if k in o else "")
            out.append(f"| {L} | {o['T_ratio']:.2f} | {o['acc1']:.3f} | {o['gap']:+.3f} | {o['composition']:+.3f}"
                       f"{ci('composition_ci')} | {o['within']:+.3f}{ci('within_ci')} | {o['shares']['cont']:.3f} | "
                       f"{o['frag_share']:.3f} | {o['cond_depth'][1]:.3f} | {o['cond_depth'][2]:.3f} |")
        out.append("\n| script group | languages | mean acc1 | mean composition | mean within |")
        out.append("|---|---|---|---|---|")
        for g, s in r["scripts"].items():
            out.append(f"| {g} | {len(s['langs'])} | {s['acc1']:.3f} | {s['composition']:+.3f} | {s['within']:+.3f} |")
    return "\n".join(out) + "\n"


def main():
    run2, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    v = json.loads((run2 / "verdict.json").read_text())
    res = {}
    for f in ("qwen", "gemma"):
        t_ratio = {L: s["T_ratio"] for L, s in v["families"][f]["langs"].items()}
        res[f] = family(run2 / f, t_ratio)
    (out / "mechanism.json").write_text(json.dumps(res, indent=1))
    (out / "MECHANISM_TABLES.md").write_text("# Mechanism tables (exploratory, run 2)\n" + tables(res), encoding="utf-8")
    print(tables(res))


if __name__ == "__main__":
    main()
