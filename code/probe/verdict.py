"""Apply the preregistered gates and kill rules to the probe outputs.

python verdict.py <out_root> [--n-items 200] [--min-langs 16]
Reads <out_root>/{qwen,gemma}/, writes <out_root>/verdict.json and <out_root>/REPORT.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fert_lib as fl  # noqa: E402

G2_MIN = 0.98
F1_MAX = 0.10
F2_MIN = 0.20
F2_MIN_TOKENS = 200
F2_MIN_LANGS = 5
R_MIN_T = 1.3
N_BOOT = 1000


def read_jsonl(p: Path):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def lang_stats(recs_L, recs_en, matched):
    by_i_L = {r["i"]: r for r in recs_L}
    by_i_E = {r["i"]: r for r in recs_en}
    L = [by_i_L[i] for i in matched]
    E = [by_i_E[i] for i in matched]
    nL, nE = sum(r["n"] for r in L), sum(r["n"] for r in E)
    s = {"n_items": len(matched), "tokens": nL, "tokens_en": nE, "T_ratio": nL / nE,
         "decode_tokens": sum(r["n"] - 1 for r in L)}  # tokens after the prefill's first one
    s["AL"], s["FP"], s["FP_en"], s["R"] = {}, {}, {}, {}
    for K in range(1, fl.K_MAX + 1):
        k = str(K)
        it = sum(r["iters"][k] for r in L)
        s["AL"][k] = sum(r["n"] - 1 for r in L) / it if it else float("nan")
        s["FP"][k] = sum(1 + r["iters"][k] for r in L)
        s["FP_en"][k] = sum(1 + r["iters"][k] for r in E)
        it_en = sum(r["iters"][k] for r in E)
        s.setdefault("AL_en", {})[k] = sum(r["n"] - 1 for r in E) / it_en if it_en else float("nan")
        if s["T_ratio"] >= R_MIN_T:
            s["R"][k] = 1 - math.log(s["FP"][k] / s["FP_en"][k]) / math.log(s["T_ratio"])
    s["ref_nll_char"] = sum(r["ref_nll"] for r in L) / sum(r["ref_chars"] for r in L)
    tag_acc = {t: [0, 0] for t in fl.TAGS}
    for r in L:
        acc = fl.decode_acc(r["acc"])
        for a in range(acc.shape[0]):  # anchor a predicts y[a+1] at depth 1
            tg = r["tags"][a + 1]
            tag_acc[tg][0] += int(acc[a, 0])
            tag_acc[tg][1] += 1
    s["acc1_by_tag"] = tag_acc
    s["acc1"] = sum(v[0] for v in tag_acc.values()) / max(1, sum(v[1] for v in tag_acc.values()))
    try:
        import sacrebleu
        s["chrF"] = sacrebleu.corpus_chrf([r["hyp"] for r in L], [[r["ref"] for r in L]]).score
    except ImportError:
        s["chrF"] = None
    return s


def item_arrays(recs_L, recs_en, matched) -> dict:
    """Per matched item (paired with its English item): the sums every per-language statistic is built from."""
    by_i_L = {r["i"]: r for r in recs_L}
    by_i_E = {r["i"]: r for r in recs_en}
    cols = {k: [] for k in ("nL", "nE", "nm1", "it3", "nll", "chars", "ws_a", "ws_c", "ct_a", "ct_c")}
    for i in matched:
        r, e = by_i_L[i], by_i_E[i]
        acc = fl.decode_acc(r["acc"])
        tags = r["tags"][1:acc.shape[0] + 1]  # anchor a predicts y[a+1] at depth 1
        a1 = acc[:, 0] if acc.shape[0] else np.zeros(0, bool)
        ws = np.array([t == "word_start" for t in tags], bool)
        ct = np.array([t == "cont" for t in tags], bool)
        for k, v in (("nL", r["n"]), ("nE", e["n"]), ("nm1", r["n"] - 1), ("it3", r["iters"]["3"]),
                     ("nll", r["ref_nll"]), ("chars", r["ref_chars"]), ("ws_a", int((a1 & ws).sum())),
                     ("ws_c", int(ws.sum())), ("ct_a", int((a1 & ct).sum())), ("ct_c", int(ct.sum()))):
            cols[k].append(v)
    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}


def item_bootstrap(arrs: dict, langs: list, d_langs: list, rng) -> tuple[list, list]:
    """Paired item bootstrap: resample matched items within each language, recompute the language
    statistics, then the cross-language F1 and F2' statistics, per replicate."""
    rs, ms = [], []
    for _ in range(N_BOOT):
        al3, tr, nll, D = [], [], [], []
        for L in langs:
            a = arrs[L]
            k = rng.integers(0, len(a["nL"]), len(a["nL"]))
            al3.append(a["nm1"][k].sum() / max(a["it3"][k].sum(), 1e-12))
            tr.append(a["nL"][k].sum() / a["nE"][k].sum())
            nll.append(a["nll"][k].sum() / a["chars"][k].sum())
            if L in d_langs:
                wc, cc = a["ws_c"][k].sum(), a["ct_c"][k].sum()
                if wc and cc:
                    D.append(a["ct_a"][k].sum() / cc - a["ws_a"][k].sum() / wc)
        if len(langs) >= 4:
            r = fl.spearman_partial(np.array(al3), np.array(tr), np.array(nll))
            if not math.isnan(r):
                rs.append(r)
        if D:
            ms.append(float(np.median(D)))
    return rs, ms


def family_eval(fdir: Path, n_items: int, min_langs: int, langs: list | None = None) -> dict:
    """langs: the language set that must be complete; the preregistered fl.LANGS unless a partial run is asked for."""
    langs = list(fl.LANGS) if langs is None else langs
    done = json.loads((fdir / "DONE.json").read_text()) if (fdir / "DONE.json").exists() else None
    g1 = json.loads((fdir / "g1.json").read_text()) if (fdir / "g1.json").exists() else None
    pf = json.loads((fdir / "preflight.json").read_text()) if (fdir / "preflight.json").exists() else None
    recs = {}
    for lang in langs:
        p = fdir / f"{lang}.jsonl"
        if p.exists():
            recs[lang] = read_jsonl(p)
    # complete = DONE.json written, no recorded size other than the expected one, and every required language
    # holding exactly items 0..n_items-1 (a smaller run or a language subset must not pass as the preregistered
    # sample; the required set comes from the caller, never from the run's own DONE.json)
    want = set(range(n_items))
    short = [L for L in langs if L not in recs or {r["i"] for r in recs[L]} != want
             or len(recs[L]) != n_items]
    out = {"complete": bool(done) and done.get("n_items", n_items) == n_items and not short,
           "incomplete_langs": short}
    out["G1"] = {"pass": bool(g1 and g1["pass"] and pf and pf["pass"]), "g1": g1,
                 "preflight": None if pf is None else {k: pf[k] for k in ("agree", "total", "rate", "pass")}}
    valid = {L: {r["i"] for r in rs if r["valid"]} for L, rs in recs.items()}
    min_valid = math.ceil(n_items / 2)
    # inclusion counts items valid in both L and en, the only items any statistic of L uses
    en_valid = valid.get("eng_Latn", set())
    matched = {L: sorted(v & en_valid) for L, v in valid.items()}
    included = [L for L in fl.LANGS if L in matched and len(matched[L]) >= min_valid]
    out["valid_counts"] = {L: len(v) for L, v in valid.items()}
    out["matched_counts"] = {L: len(v) for L, v in matched.items()}
    out["included"] = included
    m = sum(r.get("g2_match", 0) for rs in recs.values() for r in rs if r["valid"])
    t = sum(r.get("g2_total", 0) for rs in recs.values() for r in rs if r["valid"])
    out["G2"] = {"match": m, "total": t, "rate": m / t if t else float("nan"), "pass": bool(t and m / t >= G2_MIN)}
    out["G3"] = {"n_included": len(included), "pass": len(included) >= min_langs and "eng_Latn" in included}
    out["gates_pass"] = out["complete"] and out["G1"]["pass"] and out["G2"]["pass"] and out["G3"]["pass"]
    if "eng_Latn" not in included:
        return out
    stats = {L: lang_stats(recs[L], recs["eng_Latn"], matched[L]) for L in included}
    out["langs"] = stats
    al3 = np.array([stats[L]["AL"]["3"] for L in included])
    tr = np.array([stats[L]["T_ratio"] for L in included])
    nll = np.array([stats[L]["ref_nll_char"] for L in included])
    from scipy.stats import spearmanr
    rho = fl.spearman_partial(al3, tr, nll) if len(included) >= 4 else float("nan")
    out["F1"] = {"partial_spearman": rho, "raw_spearman": float(spearmanr(al3, tr).statistic)
                 if len(included) >= 3 else float("nan"), "threshold": F1_MAX,
                 "fires": bool(not math.isnan(rho) and rho <= F1_MAX)}
    ws = [L for L in included if L not in fl.NON_WHITESPACE]
    D = {}
    for L in ws:
        ta = stats[L]["acc1_by_tag"]
        if ta["word_start"][1] >= F2_MIN_TOKENS and ta["cont"][1] >= F2_MIN_TOKENS:
            D[L] = ta["cont"][0] / ta["cont"][1] - ta["word_start"][0] / ta["word_start"][1]
    med = float(np.median(list(D.values()))) if D else float("nan")
    evaluable = len(D) >= F2_MIN_LANGS
    out["F2p"] = {"D": D, "median": med, "n_langs": len(D), "evaluable": evaluable,
                  "threshold": F2_MIN, "fires": bool(evaluable and med < F2_MIN)}
    arrs = {L: item_arrays(recs[L], recs["eng_Latn"], matched[L]) for L in included}
    it_r, it_m = item_bootstrap(arrs, included, list(D), np.random.default_rng(0))
    rng = np.random.default_rng(0)
    bs_r, bs_m = [], []
    Dl = list(D.values())
    for _ in range(N_BOOT):
        k = rng.integers(0, len(included), len(included))
        r = fl.spearman_partial(al3[k], tr[k], nll[k]) if len(included) >= 4 else float("nan")
        if not math.isnan(r):
            bs_r.append(r)
        if Dl:
            bs_m.append(float(np.median(rng.choice(Dl, len(Dl)))))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None  # noqa: E731
    out["F1"]["ci95"] = ci(it_r)            # item bootstrap: the primary CI
    out["F2p"]["ci95"] = ci(it_m)
    out["F1"]["ci95_language"] = ci(bs_r)   # language bootstrap (secondary)
    out["F2p"]["ci95_language"] = ci(bs_m)
    ceil = {}
    for K in range(1, fl.K_MAX + 1):
        k = str(K)
        # English AL over the same matched items as L, so all three terms share one sample
        # "bound": the preregistered (T_L/T_en)·AL_en/(K+1), which ignores the one prefill per item and so is an
        # approximation, not a strict bound. "bound_exact": each verify step yields at most K+1 tokens, so
        # FP_L >= m + sum(n_L - 1)/(K+1); divided by the observed FP_en.
        ceil[k] = {L: {"FP_ratio": stats[L]["FP"][k] / stats[L]["FP_en"][k],
                       "bound": stats[L]["T_ratio"] * stats[L]["AL_en"][k] / (K + 1),
                       "bound_exact": (stats[L]["n_items"] + stats[L]["decode_tokens"] / (K + 1))
                       / stats[L]["FP_en"][k]} for L in included}
        for v in ceil[k].values():
            v["holds"] = bool(v["FP_ratio"] >= v["bound"] - 1e-9)
            v["holds_exact"] = bool(v["FP_ratio"] >= v["bound_exact"] - 1e-9)
    out["ceiling"] = ceil
    return out


def report(res: dict) -> str:
    lines = [f"# FERT-REFUND probe result\n\n**Verdict: {res['verdict']}**  ({res['reason']})\n"]
    lines += [f"- note: {n}" for n in res.get("notes", [])]
    for f, r in res["families"].items():
        lines.append(f"\n## {f}\n")
        if r is None:
            lines.append("no output\n")
            continue
        g1 = r["G1"]["g1"] or {}
        lines.append(f"- complete: {r['complete']}; G1 pass: {r['G1']['pass']} "
                     f"(rates {json.dumps(g1.get('rate'))}, primary {g1.get('primary_wiring')}, "
                     f"preflight {r['G1']['preflight']}); G2 {r['G2']['rate']:.4f} pass {r['G2']['pass']}; "
                     f"G3 {r['G3']['n_included']} langs pass {r['G3']['pass']}")
        if "F1" in r:
            lines.append(f"- F1: partial Spearman {r['F1']['partial_spearman']:.3f} (item-bootstrap CI {r['F1']['ci95']}), "
                         f"raw {r['F1']['raw_spearman']:.3f}; fires {r['F1']['fires']}")
            lines.append(f"- F2': median D {r['F2p']['median']:.3f} over {r['F2p']['n_langs']} langs "
                         f"(item-bootstrap CI {r['F2p']['ci95']}); evaluable {r['F2p']['evaluable']}; fires {r['F2p']['fires']}\n")
            lines.append("| lang | items | T/T_en | AL K=1 | AL K=3 | AL K=5 | R K=3 | acc1 | acc1 ws | acc1 cont | "
                         "nll/char | chrF |")
            lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for L, s in r["langs"].items():
                ta = s["acc1_by_tag"]
                fr = lambda t: f"{ta[t][0] / ta[t][1]:.3f}" if ta[t][1] else "-"  # noqa: E731
                R3 = "-" if s["R"].get("3") is None else "%.3f" % s["R"]["3"]
                chrf = "-" if s["chrF"] is None else "%.1f" % s["chrF"]
                lines.append(f"| {L} | {s['n_items']} | {s['T_ratio']:.2f} | {s['AL']['1']:.3f} | {s['AL']['3']:.3f} | "
                             f"{s['AL']['5']:.3f} | {R3} | {s['acc1']:.3f} | "
                             f"{fr('word_start')} | {fr('cont')} | {s['ref_nll_char']:.3f} | {chrf} |")
    return "\n".join(lines) + "\n"


# the preregistered design; any other value makes the verdict PARTIAL
PREREG_VERDICT = {"n_items": 200, "min_langs": 16, "langs": "all"}
PREREG_RUN = {"langs": "all", "n_items": 200, "g1_items": 20, "batch": 50, "max_new": 400, "dtype": "bfloat16",
              "no_abort": False, "no_cap_filter": False}
PREREG_LIB = {"transformers": "5.19.0"}
# run 2 (Colab, 2026-10-09) pre-dates recorded revisions and fingerprint.json; its two family directories, identified
# by family_manifest over every run file (JSONL, gates, meta, DONE), are the only runs whose provenance rests on the
# model revisions pinned in the notebook that produced them (it sha256-checks its embedded code and the FLORES
# tarball) instead. Any other run must record revisions and a fingerprint.
ATTESTED_LEGACY_FAMILIES = {"qwen": "ec52681a3bb88296904e88a158fe172a15558455999bbe621112977eb5d9a923",
                            "gemma": "33675d5da8d2416059ba31ebe7c3eee802a95b2adddcde260586ab2f5a96c677"}


def family_manifest(fdir: Path) -> str:
    """sha256 over the names and sha256 of every run file in a family directory."""
    h = hashlib.sha256()
    for p in sorted(fdir.iterdir()):
        if p.is_file() and (p.suffix == ".jsonl" or p.name in ("DONE.json", "g1.json", "preflight.json", "meta.json")):
            h.update(f"{p.name}\0{hashlib.sha256(p.read_bytes()).hexdigest()}\0".encode())
    return h.hexdigest()
# fert_probe.dir_manifest of the devtest dir extracted from the sha256-pinned FLORES-200 tarball (make_notebook.py)
PREREG_FLORES_MANIFEST = "98bf69f75e6ac396e4de8a5de93d975a1d3005f9bfe7b750a82e24fe96a79ea9"
PREREG_REVISIONS = {"qwen": {"target": "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "drafter": None},
                    "gemma": {"target": "ee0ef6023621cff504d758262d4e04895a5af4a2",
                              "drafter": "8d0031ea8c2109e2b1e86bb9368a4539b537f80a"}}


def deviations(args, root: Path, notes: list | None = None) -> list[str]:
    """Settings of this verdict call and of each family's recorded run (meta.json) that differ from the design.
    A revision that is not recorded, missing or different is a deviation, except for the attested run 2 meta.json
    files (ATTESTED_LEGACY_META), which get a note."""
    dev = [f"verdict --{k.replace('_', '-')} {getattr(args, k)!r} != {v!r}"
           for k, v in PREREG_VERDICT.items() if getattr(args, k) != v]
    for f in fl.FAMILIES:
        mp = root / f / "meta.json"
        if not mp.exists():
            dev.append(f"{f}: no meta.json")
            continue
        meta = json.loads(mp.read_text())
        run = meta["args"]
        dev += [f"{f}: --{k.replace('_', '-')} {run.get(k)!r} != {v!r}" for k, v in PREREG_RUN.items()
                if run.get(k) != v]
        dev += [f"{f}: {k} {meta.get(k)!r} != {v!r}" for k, v in PREREG_LIB.items() if meta.get(k) != v]
        legacy = family_manifest(root / f) == ATTESTED_LEGACY_FAMILIES[f]
        if meta.get("args", {}).get("device") != "cuda" or "A100" not in str(meta.get("gpu")):
            dev.append(f"{f}: ran on {meta.get('args', {}).get('device')!r} / {meta.get('gpu')!r}, not one A100")
        if legacy and notes is not None:
            notes.append(f"{f}: run 2 pre-dates recorded revisions and fingerprint.json; its models, code and FLORES "
                         "rest on the sha-checked notebook pins")
        if "revisions" in meta:
            dev += [f"{f}: {k} revision {meta['revisions'].get(k)!r} != {v!r}"
                    for k, v in PREREG_REVISIONS[f].items() if meta["revisions"].get(k) != v]
        elif not legacy:
            dev.append(f"{f}: model revisions not recorded")
        # the run must use the canonical FLORES text and the same probe code as this verdict's checkout
        fpp = root / f / "fingerprint.json"
        if fpp.exists():
            fp = json.loads(fpp.read_text())
            here = Path(__file__).resolve().parent
            dev += [f"{f}: {name} differs from this checkout" for name, h in fp.get("code", {}).items()
                    if hashlib.sha256((here / name).read_bytes()).hexdigest() != h]
            if fp.get("flores") != PREREG_FLORES_MANIFEST:
                dev.append(f"{f}: FLORES directory is not the canonical devtest")
        elif not legacy:
            dev.append(f"{f}: no fingerprint.json")
    return dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_root")
    ap.add_argument("--n-items", type=int, default=200)
    ap.add_argument("--min-langs", type=int, default=16)
    ap.add_argument("--langs", default="all", help="SMOKE/partial only: a subset (any non-preregistered setting "
                    "makes the verdict PARTIAL)")
    args = ap.parse_args()
    root = Path(args.out_root)
    langs = list(fl.LANGS) if args.langs == "all" else args.langs.split(",")
    fam = {f: (family_eval(root / f, args.n_items, args.min_langs, langs) if (root / f).exists() else None)
           for f in fl.FAMILIES}
    if any(r is None or not r["gates_pass"] for r in fam.values()):
        bad = [f for f, r in fam.items() if r is None or not r["gates_pass"]]
        verdict, reason = "INVALID", f"gate failure or incomplete run in {bad}"
    elif all(r["F1"]["fires"] for r in fam.values()):
        verdict, reason = "KILLED", "F1 fires in both families"
    elif all(r["F2p"]["fires"] for r in fam.values()):
        verdict, reason = "KILLED", "F2' fires in both families"
    else:
        verdict, reason = "SURVIVES", "no kill rule fires in both families"
    notes = []
    dev = deviations(args, root, notes)
    if dev:
        verdict, reason = f"PARTIAL ({verdict})", f"{reason}; not the preregistered design: {'; '.join(dev)}"
    res = {"verdict": verdict, "reason": reason, "deviations": dev, "notes": notes, "families": fam}
    (root / "verdict.json").write_text(json.dumps(res, indent=1, default=float))
    (root / "REPORT.md").write_text(report(res), encoding="utf-8")
    print(report(res))
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
