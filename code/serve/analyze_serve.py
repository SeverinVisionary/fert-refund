"""Analyses A1-A3 of PREREG_SERVE.md over the fert_serve.py outputs.

python analyze_serve.py <out dir> [--run2 <unzipped run 2 dir>]  -> <out>/serve_summary.json, SERVE_REPORT.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parent.parent / "probe"))  # fert_lib in the repo
import fert_lib as fl  # noqa: E402
import fert_serve as fs  # noqa: E402

MTP_ARMS = ["qwen", "gemma"]
EAGLE_ARMS = ["eagle_rh", "eagle_patel"]
LAT_CONFIGS = {"qwen": [0, 1, 3, 5], "gemma": [0, 1, 3, 5], "eagle_rh": [3], "eagle_patel": [3]}
BASE = {"qwen": "qwen_K0", "gemma": "gemma_K0", "eagle_rh": "qwen3_8b_K0", "eagle_patel": "qwen3_8b_K0"}
N_BOOT = 2000


def read_jsonl(p: Path):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else None


def valid(y, finish) -> bool:
    """Run 2's rule: not capped at the token limit, 4-gram repeat fraction <= 0.3."""
    return finish != "length" and len(y) < fs.MAX_NEW and fl.rep4_fraction(y) <= 0.3


def spearman(x, y) -> float:
    def rank(v):
        v = np.asarray(v, float)
        o = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        r[o] = np.arange(len(v))
        for u in np.unique(v):  # average ties
            m = v == u
            r[m] = r[m].mean()
        return r
    if len(x) < 3:
        return float("nan")
    a, b = rank(x), rank(y)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def boot_ratio(num, den, seed=0):
    """Item-bootstrap 95% CI of sum(num)/sum(den) over paired items."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    if len(num) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(N_BOOT, len(num)))
    r = num[idx].sum(1) / den[idx].sum(1)
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def latency_eval(out: Path) -> dict:
    lat = {}
    for tag in {f"{a}_K{k}" for a, ks in LAT_CONFIGS.items() for k in ks} | set(BASE.values()):
        recs = read_jsonl(out / f"{tag}_latency.jsonl")
        if recs is not None:
            lat[tag] = {(r["lang"], r["i"]): r for r in recs}
    res = {}
    for arm, ks in LAT_CONFIGS.items():
        base = lat.get(BASE[arm])
        if base is None:
            continue
        for k in ks:
            cfg = lat.get(f"{arm}_K{k}")
            if cfg is None:
                continue
            res.setdefault(arm, {})
            per = {}
            for lang in fs.LAT_LANGS:
                keys = [key for key in base if key[0] == lang and key in cfg
                        and valid(base[key]["y"], base[key]["finish"]) and valid(cfg[key]["y"], cfg[key]["finish"])]
                if not keys:
                    continue
                lb = [base[x]["lat"] for x in keys]
                lk = [cfg[x]["lat"] for x in keys]
                ntok = [len(cfg[x]["y"]) for x in keys]
                dr = sum(cfg[x]["drafts"] for x in keys)
                per[lang] = {"n": len(keys), "speedup": sum(lb) / sum(lk), "speedup_ci": boot_ratio(lb, lk),
                             "lat_ms": 1000 * float(np.mean(lk)), "lat_none_ms": 1000 * float(np.mean(lb)),
                             "ms_per_tok": 1000 * sum(lk) / sum(ntok),
                             "al": 1 + sum(cfg[x]["accepted"] for x in keys) / dr if dr else float("nan"),
                             "al_ci": [1 + v for v in boot_ratio([cfg[x]["accepted"] for x in keys],
                                                                 [cfg[x]["drafts"] for x in keys])] if dr else None,
                             "lossless": float(np.mean([cfg[x]["y"] == base[x]["y"] for x in keys])),
                             "_keys": keys}
            en = per.get("eng_Latn")
            for lang, s in per.items():
                if en:
                    s["premium"] = s["lat_ms"] / en["lat_ms"]
                    s["premium_none"] = s["lat_none_ms"] / en["lat_none_ms"]
                del s["_keys"]
            res[arm][str(k)] = per
    return res


def sampling_eval(out: Path) -> dict:
    res = {}
    for arm in MTP_ARMS + EAGLE_ARMS:
        recs = read_jsonl(out / f"{arm}_K3_sampling.jsonl")
        if recs is None:
            continue
        by = {(r["temp"], r["lang"]): r for r in recs}
        # T_L/T_en: the arm's own greedy output tokens over items valid in both L and English
        g_en = by.get((0.0, "eng_Latn"))
        res[arm] = {"T_ratio": {}, "AL": {}, "rho": {}}
        if g_en is None:
            continue
        en_ok = {it["i"]: len(it["y"]) for it in g_en["items"] if valid(it["y"], it["finish"])}
        for (t, lang), r in by.items():
            if t != 0.0:
                continue
            ok = {it["i"]: len(it["y"]) for it in r["items"] if valid(it["y"], it["finish"])}
            m = sorted(set(ok) & set(en_ok))
            if m:
                res[arm]["T_ratio"][lang] = sum(ok[i] for i in m) / sum(en_ok[i] for i in m)
        for t in fs.TEMPS:
            als = {lang: (1 + r["accepted"] / r["drafts"]) if r["drafts"] else float("nan")
                   for (tt, lang), r in by.items() if tt == t}
            res[arm]["AL"][str(t)] = als
            langs = [L for L in als if L in res[arm]["T_ratio"] and not math.isnan(als[L])]
            res[arm]["rho"][str(t)] = spearman([als[L] for L in langs], [res[arm]["T_ratio"][L] for L in langs])
    return res


def calib_eval(out: Path, run2: Path | None) -> dict:
    res = {}
    for arm in ["qwen", "gemma", "eagle_rh", "eagle_patel"]:
        p = out / f"{arm}_K3_calib.json"
        if not p.exists():
            res[arm] = None
            continue
        res[arm] = json.loads(p.read_text())
        recs = read_jsonl(out / f"{arm}_K3_calib.jsonl")
        if run2 and arm in MTP_ARMS and recs and (run2 / arm / "eng_Latn.jsonl").exists():
            ref = {r["i"]: r["y"] for r in read_jsonl(run2 / arm / "eng_Latn.jsonl")}
            res[arm]["same_as_run2_hf"] = float(np.mean([r["y"] == ref.get(r["i"]) for r in recs]))
    return res


def report(s: dict) -> str:
    L = ["# FERT-REFUND serving extension: results\n"]
    L.append("## Calibration\n")
    for arm, c in s["calib"].items():
        L.append(f"- {arm}: {json.dumps(c) if c else 'not run'}")
    L.append("\n## A1 latency (batch 1)\n")
    for arm, ks in s["latency"].items():
        for k, per in ks.items():
            L.append(f"\n### {arm} K={k}\n")
            L.append("| lang | n | speedup (CI) | ms | ms none | ms/token | AL | premium | premium none | lossless |")
            L.append("|---|---|---|---|---|---|---|---|---|---|")
            for lang, v in per.items():
                L.append(f"| {lang} | {v['n']} | {v['speedup']:.3f} [{v['speedup_ci'][0]:.3f}, {v['speedup_ci'][1]:.3f}] | "
                         f"{v['lat_ms']:.0f} | {v['lat_none_ms']:.0f} | {v['ms_per_tok']:.2f} | {v['al']:.3f} | "
                         f"{v.get('premium', float('nan')):.3f} | {v.get('premium_none', float('nan')):.3f} | "
                         f"{v['lossless']:.2f} |")
    L.append("\n## A2/A3 sampling, K=3\n")
    for arm, r in s["sampling"].items():
        L.append(f"- {arm}: Spearman(AL, T/T_en) by temperature {json.dumps(r['rho'])}")
    L.append("\n| arm | lang | T/T_en | AL T=0 | AL T=0.7 | AL T=1 |")
    L.append("|---|---|---|---|---|---|")
    for arm, r in s["sampling"].items():
        for lang in fl.LANGS:
            if lang in r["T_ratio"]:
                a = [r["AL"].get(str(t), {}).get(lang, float("nan")) for t in fs.TEMPS]
                L.append(f"| {arm} | {lang} | {r['T_ratio'][lang]:.2f} | {a[0]:.3f} | {a[1]:.3f} | {a[2]:.3f} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--run2")
    args = ap.parse_args(argv)
    out = Path(args.out)
    s = {"calib": calib_eval(out, Path(args.run2) if args.run2 else None),
         "latency": latency_eval(out), "sampling": sampling_eval(out)}
    (out / "serve_summary.json").write_text(json.dumps(s, indent=1, default=float))
    (out / "SERVE_REPORT.md").write_text(report(s), encoding="utf-8")
    print(report(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
