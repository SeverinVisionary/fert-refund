"""PREREG_SERVE.md analyses that analyze_serve.py (the version that ran) does not emit. Added after the data.

python analyze_serve_extra.py <serving out dir> <unzipped run 2 dir>  -> <out>/serve_extra.json

- A1: Spearman rho(S_L(3), T_L/T_en) over the six latency languages, per arm. T_L/T_en is the arm's own
  T = 0 sampling token ratio (A2's definition), restricted to those languages.
- A3: per-language T = 0 AL of eagle_patel minus eagle_rh.
- Reported only: the fraction of qwen/gemma English calibration outputs equal to run 2's HF greedy output.
- Sensitivity (not preregistered): sampling AL comes from per-language counters, so it cannot exclude invalid
  items. Per arm and T, report the invalid count per language, and A2's rho again without any language whose
  sampling items are more than 10% invalid; and the same rho over exclusion thresholds of 5, 10, 20, 30 and 50%,
  because the 10% result is the largest of them for eagle_patel.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import analyze_serve as an
import fert_serve as fs


def extra(out: Path, run2: Path) -> dict:
    s = {"latency": an.latency_eval(out), "sampling": an.sampling_eval(out), "calib": an.calib_eval(out, run2)}
    rho = {}
    for arm, by_k in s["latency"].items():
        lat, ratio = by_k.get("3"), s["sampling"].get(arm, {}).get("T_ratio")
        if lat and ratio:
            langs = [L for L in fs.LAT_LANGS if L in lat and L in ratio]
            rho[arm] = {"n": len(langs), "rho": an.spearman([lat[L]["speedup"] for L in langs],
                                                            [ratio[L] for L in langs])}
    al = {a: s["sampling"].get(a, {}).get("AL", {}).get("0.0") for a in an.EAGLE_ARMS}
    diff = {L: al["eagle_patel"][L] - al["eagle_rh"][L] for L in al["eagle_rh"]} if all(al.values()) else None
    same = {a: (s["calib"].get(a) or {}).get("same_as_run2_hf") for a in an.MTP_ARMS}
    sens = {}
    for arm, sm in s["sampling"].items():
        recs = an.read_jsonl(out / f"{arm}_K3_sampling.jsonl") or []
        bad = {(str(r["temp"]), r["lang"]): sum(not an.valid(it["y"], it.get("finish")) for it in r["items"])
               / len(r["items"]) for r in recs}
        sens[arm] = {}
        for t, al_t in sm["AL"].items():
            keep = [L for L in al_t if L in sm["T_ratio"] and bad.get((t, L), 0) <= 0.10]
            sens[arm][t] = {"dropped": sorted(L for L in al_t if L not in keep),
                            "invalid_frac": {L: bad[(t, L)] for L in al_t if bad.get((t, L), 0) > 0},
                            "rho_kept": an.spearman([al_t[L] for L in keep], [sm["T_ratio"][L] for L in keep]),
                            "n_kept": len(keep)}
    return {"A1_rho_speedup_vs_T_ratio": rho, "A3_patel_minus_rh_AL_T0": diff, "calib_same_as_run2_hf": same,
            "sensitivity_A2_drop_langs_over_10pct_invalid": sens,
            "sensitivity_A2_threshold_sweep": {
                arm: {str(thr): {t: (lambda keep: {"rho": an.spearman([al_t[L] for L in keep],
                                                                      [sm["T_ratio"][L] for L in keep]), "n": len(keep)})(
                    [L for L in al_t if L in sm["T_ratio"] and sens[arm][t]["invalid_frac"].get(L, 0) <= thr])
                    for t, al_t in sm["AL"].items()} for thr in (0.05, 0.10, 0.20, 0.30, 0.50)}
                for arm, sm in s["sampling"].items()}}


if __name__ == "__main__":
    out = Path(sys.argv[1])
    res = extra(out, Path(sys.argv[2]))
    (out / "serve_extra.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
