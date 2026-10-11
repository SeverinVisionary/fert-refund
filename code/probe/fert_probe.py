"""FERT-REFUND probe runner, one model family per invocation.

Stages (all outputs under <out>/<family>/):
  g1.json        wiring gate G1 on the first --g1-items English items, plus the Gemma position
                 convention choice (HF t+1 vs vLLM t), decided on English step-1 acceptance only
  preflight.json batched emulation vs the sequential reference on real weights (2 items)
  <lang>.jsonl   one record per item; a language is skipped when its file is complete (resume)
  DONE.json      timing
  fingerprint.json  probe code, settings and library versions; an --out holding a different run is refused
Exit code 3 = a pre-run check failed and the bulk run was not started (unless --no-abort).

python fert_probe.py --family qwen --target DIR --flores DIR --out DIR
python fert_probe.py --family gemma --target DIR --drafter DIR --flores DIR --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fert_lib as fl  # noqa: E402

G1_MIN_ACC = 0.50
G1_MARGIN = 0.05
POS_SWITCH_MARGIN = 0.02
PREFLIGHT_MIN_AGREE = 0.97
REP4_MAX = 0.3


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load(family: str, args):
    from transformers import AutoTokenizer
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.target)
    if family == "qwen":
        from transformers import Qwen3_5ForConditionalGeneration as Cls
    else:
        from transformers import Gemma4ForConditionalGeneration as Cls
    model = Cls.from_pretrained(args.target, dtype=dtype, attn_implementation="sdpa").to(args.device).eval()
    if family == "qwen":
        drafter = fl.load_qwen_mtp(args.target, model.config.text_config, dtype, args.device)
    else:
        from transformers import Gemma4AssistantForCausalLM
        drafter = Gemma4AssistantForCausalLM.from_pretrained(
            args.drafter, dtype=dtype, attn_implementation="sdpa").to(args.device).eval()
        hb = model.config.get_text_config().hidden_size
        if drafter.config.backbone_hidden_size != hb:
            raise ValueError(f"assistant backbone {drafter.config.backbone_hidden_size} != target {hb}")
    return tok, model, drafter


def write_jsonl(path: Path, recs: list[dict]):
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def generate(model, prompts, args, stops, pad_id):
    order = sorted(range(len(prompts)), key=lambda k: -len(prompts[k]))
    out = [None] * len(prompts)
    for b0 in range(0, len(order), args.batch):
        idx = order[b0:b0 + args.batch]
        res = fl.generate_batch(model, [prompts[k] for k in idx], args.max_new, stops, pad_id, args.device)
        for k, r in zip(idx, res):
            out[k] = r
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    return out


def g1_stage(family, tok, model, drafter, flores, args, stops, pad_id, outdir: Path) -> dict:
    spec = fl.FAMILIES[family]
    n = min(args.g1_items, args.n_items)
    prompts = [fl.chat_prompt_ids(tok, fl.make_prompt("eng_Latn", i, flores)[0]) for i in range(n)]
    gens = generate(model, prompts, args, stops, pad_id)
    counts = {w: [0, 0] for w in spec["wirings"]}
    items = []
    for p, (y, capped) in zip(prompts, gens):
        if capped or len(y) < 2:
            continue
        ids = torch.tensor(p + y, device=args.device)
        tf = None
        for w in spec["wirings"]:
            d, tf = fl.item_drafts(family, model, drafter, ids, len(p), w, tf)
            a, t = fl.step1_acceptance(fl.acceptance_matrix(d, ids, len(p)))
            counts[w][0] += a
            counts[w][1] += t
        items.append((p, y))
    rate = {w: (c[0] / c[1] if c[1] else float("nan")) for w, c in counts.items()}
    primary = "ok"
    if family == "gemma" and rate["pos_t"] >= rate["ok"] + POS_SWITCH_MARGIN:
        primary = "pos_t"
    reasons = []
    if len(items) < n:  # the preregistered check is over all of the first n English items
        reasons.append(f"only {len(items)} of {n} G1 items usable (capped or < 2 tokens)")
    if not rate[primary] >= G1_MIN_ACC:
        reasons.append(f"primary step-1 acceptance {rate[primary]:.3f} < {G1_MIN_ACC}")
    for m in spec["miswirings"]:
        w = fl.compose_wiring(primary, m)  # same position convention as the primary
        if not rate[primary] - rate[w] >= G1_MARGIN:
            reasons.append(f"primary beats {w} by {rate[primary] - rate[w]:.3f} < {G1_MARGIN}")
    res = {"family": family, "n_items": n, "n_used": len(items), "counts": counts, "rate": rate,
           "primary_wiring": primary, "pass": not reasons, "reasons": reasons}
    (outdir / "g1.json").write_text(json.dumps(res, indent=1))
    log("G1", json.dumps(res))
    return res, items


def preflight(family, model, drafter, items, wiring, args, outdir: Path) -> dict:
    agree = total = 0
    detail = []
    for p, y in items[:2]:
        ids = torch.tensor(p + y, device=args.device)
        P = len(p)
        d, tf = fl.item_drafts(family, model, drafter, ids, P, wiring)
        T = d.shape[0]
        for a in sorted(set(np.linspace(0, T - 1, min(T, 16)).astype(int).tolist())):
            t = P - 1 + a
            if family == "qwen":
                ref = fl.qwen_reference_drafts(drafter, model, ids, fl.qwen_hidden(tf, wiring), t, fl.K_MAX,
                                               "swap" if wiring == "swap" else "ok")
            else:
                ref = fl.gemma_reference_drafts(drafter, model, ids, tf["h"], tf["kv"], t, fl.K_MAX, wiring)
            got = d[a].tolist()
            agree += sum(int(x == r) for x, r in zip(got, ref))
            total += len(ref)
            detail.append({"t": t, "batched": got, "reference": ref})
    rate = agree / total if total else float("nan")
    res = {"agree": agree, "total": total, "rate": rate, "pass": bool(rate >= PREFLIGHT_MIN_AGREE),
           "threshold": PREFLIGHT_MIN_AGREE, "detail": detail}
    (outdir / "preflight.json").write_text(json.dumps(res, indent=1))
    log(f"preflight agreement {agree}/{total} = {rate:.4f}")
    return res


def process(family, tok, model, drafter, tb, wiring, lang, i, prompt, y, capped, ref, args):
    n = len(y)
    rep = fl.rep4_fraction(y)
    valid = (not capped or args.no_cap_filter) and rep <= REP4_MAX
    rec = {"lang": lang, "i": i, "P": len(prompt), "n": n, "capped": capped, "rep4": rep,
           "valid": valid, "y": y, "hyp": tok.decode(y, skip_special_tokens=True).strip(), "ref": ref}
    if not valid:
        return rec
    ids = torch.tensor(prompt + y, device=args.device)
    P, N = len(prompt), len(prompt) + n
    d, tf = fl.item_drafts(family, model, drafter, ids, P, wiring)
    acc = fl.acceptance_matrix(d, ids, P)
    rec["acc"] = fl.encode_acc(acc)
    rec["iters"] = {str(K): fl.simulate(acc, P, N, K) for K in range(1, fl.K_MAX + 1)}
    rec["tags"] = fl.tag_tokens(y, tb)
    am = tf["argmax"][P - 1:N - 1].cpu()
    rec["g2_match"] = int((am == ids[P:].cpu()).sum())
    rec["g2_total"] = n
    ref_ids = tok(ref, add_special_tokens=False)["input_ids"]
    rec["ref_nll"] = fl.ref_nll(model, family, prompt, ref_ids, args.device)
    rec["ref_chars"] = len(ref)
    rec["ref_ntok"] = len(ref_ids)
    return rec


def run_fingerprint(args) -> dict:
    """Everything that can change a measurement: probe code, settings, model paths, library versions."""
    import transformers
    here = Path(__file__).resolve().parent
    code = {f: hashlib.sha256((here / f).read_bytes()).hexdigest() for f in ("fert_lib.py", "fert_probe.py")}
    settings = {k: v for k, v in vars(args).items() if k != "out"}
    models = {k: dir_manifest(getattr(args, k)) for k in ("target", "drafter")}
    return {"code": code, "args": settings, "models": models, "flores": dir_manifest(args.flores),
            "torch": torch.__version__,
            "transformers": transformers.__version__}


def dir_manifest(path) -> str | None:
    """Hash of a dir's file names and sizes plus every small file (HF download metadata with commit hash and etag,
    configs, FLORES text), so a different revision or dataset at the same path changes the fingerprint without
    hashing weights."""
    if not path or not Path(path).is_dir():
        return None
    h = hashlib.sha256()
    for f in sorted(p for p in Path(path).rglob("*") if p.is_file()):
        rel, size = f.relative_to(path).as_posix(), f.stat().st_size
        h.update(f"{rel}\0{size}\0".encode())
        if size <= 1 << 20 and not rel.endswith(".lock"):
            h.update(f.read_bytes())
        else:  # large files (weights): an in-place rewrite changes mtime; the HF etag above covers downloads
            h.update(f"{f.stat().st_mtime_ns}\0".encode())
    return h.hexdigest()


def check_outdir(outdir: Path, fp: dict, langs: list, n_items: int) -> bool:
    """Refuse an --out from a different run (exit 4); True when this run already finished there."""
    path = outdir / "fingerprint.json"
    if path.exists():
        prev = json.loads(path.read_text())
        if prev != fp:
            diff = sorted(k for k in set(prev) | set(fp) if prev.get(k) != fp.get(k))
            log(f"{outdir} holds a run with a different configuration (differs in {diff}); use a new --out")
            sys.exit(4)
    elif any(outdir.iterdir()):
        log(f"{outdir} holds files but no fingerprint.json; use a new --out")
        sys.exit(4)
    else:
        path.write_text(json.dumps(fp, indent=1))
    done = outdir / "DONE.json"
    if not done.exists():
        return False
    d = json.loads(done.read_text())
    return (d.get("n_items") == n_items and d.get("langs") == langs
            and all(len(read_jsonl(outdir / f"{L}.jsonl")) == n_items for L in langs))


def model_revision(path) -> str | None:
    """Commit hash of a model dir: an HF cache snapshot path, or the HF local_dir download metadata, where every
    file's metadata must name the same commit (a dir mixing revisions records "mixed:...")."""
    if not path:
        return None
    p = Path(path)
    if p.parent.name == "snapshots" and len(p.name) == 40:
        return p.name
    revs = {(m.read_text().splitlines() or [""])[0].strip()
            for m in (p / ".cache/huggingface/download").rglob("*.metadata") if m.is_file()}
    if len(revs) > 1:
        return "mixed:" + ",".join(sorted(revs))
    rev = revs.pop() if revs else ""
    return rev or None


def main():
    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1)  # `kill -USR1 <pid>` prints every thread's stack to stderr
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(fl.FAMILIES))
    ap.add_argument("--target", required=True)
    ap.add_argument("--drafter")
    ap.add_argument("--flores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", default="all")
    ap.add_argument("--n-items", type=int, default=200)
    ap.add_argument("--g1-items", type=int, default=20)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--no-abort", action="store_true", help="SMOKE only: continue past failed checks")
    ap.add_argument("--no-cap-filter", action="store_true", help="SMOKE only: keep capped items")
    args = ap.parse_args()
    family = args.family
    if family == "gemma" and not args.drafter:
        ap.error("--drafter is required for gemma")
    langs = fl.LANGS if args.langs == "all" else args.langs.split(",")
    if langs[0] != "eng_Latn":
        langs = ["eng_Latn"] + [x for x in langs if x != "eng_Latn"]
    outdir = Path(args.out) / family
    outdir.mkdir(parents=True, exist_ok=True)
    if check_outdir(outdir, run_fingerprint(args), langs, args.n_items):
        log(f"{family}: already complete in {outdir}, nothing to do")
        return
    torch.manual_seed(0)
    t_start = time.time()

    tok, model, drafter = load(family, args)
    stops = fl.stop_ids(tok, family)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else stops[-1]
    tb = fl.TokenBytes(tok)
    flores = fl.load_flores(args.flores, langs, args.n_items)
    import transformers
    meta = {"family": family, "spec": fl.FAMILIES[family], "args": vars(args), "langs": langs,
            "stops": stops, "pad_id": pad_id, "torch": torch.__version__,
            "transformers": transformers.__version__, "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "revisions": {"target": model_revision(args.target), "drafter": model_revision(args.drafter)},
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (outdir / "meta.json").write_text(json.dumps(meta, indent=1))
    log("loaded", family, "stops", stops, "pad", pad_id)

    g1_path = outdir / "g1.json"
    g1_prev = json.loads(g1_path.read_text()) if g1_path.exists() else None
    if (g1_prev and set(g1_prev["counts"]) == set(fl.FAMILIES[family]["wirings"])
            and (outdir / "preflight.json").exists()):
        g1 = g1_prev  # resume: same fingerprint, so same code, settings and wiring set
        pf = json.loads((outdir / "preflight.json").read_text())
        log("reusing g1.json / preflight.json")
    else:
        g1, items = g1_stage(family, tok, model, drafter, flores, args, stops, pad_id, outdir)
        pf = preflight(family, model, drafter, items, g1["primary_wiring"], args, outdir)
    if not (g1["pass"] and pf["pass"]):
        log("PRE-RUN CHECK FAILED", g1["reasons"], "preflight", pf["rate"])
        if not args.no_abort:
            sys.exit(3)
    wiring = g1["primary_wiring"]

    for lang in langs:
        path = outdir / f"{lang}.jsonl"
        done = read_jsonl(path)
        if len(done) == args.n_items:
            log(f"{lang}: complete, skipping")
            continue
        t0 = time.time()
        prompts, refs = [], []
        for i in range(args.n_items):
            msg, ref = fl.make_prompt(lang, i, flores)
            prompts.append(fl.chat_prompt_ids(tok, msg))
            refs.append(ref)
        gens = generate(model, prompts, args, stops, pad_id)
        t1 = time.time()
        recs = [process(family, tok, model, drafter, tb, wiring, lang, i, prompts[i], y, c, refs[i], args)
                for i, (y, c) in enumerate(gens)]
        write_jsonl(path, recs)
        nv = sum(r["valid"] for r in recs)
        al3 = sum(r["n"] - 1 for r in recs if r["valid"]) / max(1, sum(r["iters"]["3"] for r in recs if r["valid"]))
        log(f"{lang}: valid {nv}/{len(recs)}  AL(K=3) {al3:.3f}  gen {t1 - t0:.0f}s  emul {time.time() - t1:.0f}s")
    (outdir / "DONE.json").write_text(json.dumps({"seconds": time.time() - t_start, "n_items": args.n_items,
                                                   "langs": langs,
                                                   "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}))
    log("done", family, f"{time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
