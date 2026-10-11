#!/usr/bin/env bash
# Recompute the verdict and the mechanism decomposition from data/ and check them against the shipped files.
# Every step fails loudly; exit status 0 means everything matched.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
"$PY" -c "import numpy, scipy, sacrebleu, torch, transformers" || { echo "missing dependency (see README.md)"; exit 1; }

echo "1/3 checksums"
grep -v '  SHA256SUMS$' SHA256SUMS | (command -v sha256sum >/dev/null && sha256sum -c --quiet - || shasum -a 256 -c --quiet -)

WORK=$(mktemp -d "${TMPDIR:-.}/fert_check.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
mkdir "$WORK/run2" && cp -R data/run2/qwen data/run2/gemma "$WORK/run2/"   # no shipped verdict.json in the copy

echo "2/3 verdict and gates"
"$PY" code/probe/verdict.py "$WORK/run2" > "$WORK/verdict.log"
test -s "$WORK/run2/verdict.json" || { echo "verdict.py wrote no verdict.json"; exit 1; }
"$PY" - "$WORK/run2/verdict.json" data/run2/verdict.json data/run2 <<'PYEOF'
import json, math, sys
sys.path.insert(0, "code/probe")
import fert_lib as fl
new, old = (json.load(open(p)) for p in sys.argv[1:3])
assert new["verdict"] == old["verdict"] == "KILLED", (new["verdict"], old["verdict"])
assert new["reason"] == old["reason"], (new["reason"], old["reason"])
assert new.get("deviations") == [], new.get("deviations")
n = 0
def sub(o, x, path):
    """every field of the shipped verdict must be reproduced (the newer code may add fields)"""
    global n
    if isinstance(o, dict):
        assert isinstance(x, dict), path
        for k in o:
            assert k in x, ("missing", path, k)
            sub(o[k], x[k], f"{path}.{k}")
    elif isinstance(o, list):
        assert isinstance(x, list) and len(o) == len(x), path
        for i, (a, b) in enumerate(zip(o, x)):
            sub(a, b, f"{path}[{i}]")
    elif isinstance(o, float) and not isinstance(o, bool):
        n += 1
        assert (math.isnan(o) and math.isnan(x)) or abs(o - x) <= 1e-9 * max(1.0, abs(o)), (path, o, x)
    else:
        n += 1
        assert o == x, (path, o, x)
sub(old["families"], new["families"], "$")
for f in ("qwen", "gemma"):   # recompute G1 and preflight from their stored counts
    g1 = json.load(open(f"{sys.argv[3]}/{f}/g1.json"))
    pf = json.load(open(f"{sys.argv[3]}/{f}/preflight.json"))
    spec = json.load(open(f"{sys.argv[3]}/{f}/meta.json"))["spec"]
    rate = {w: a / t for w, (a, t) in g1["counts"].items()}
    p = g1["primary_wiring"]
    ok = (g1["n_used"] == g1["n_items"] == 20 and rate[p] >= 0.50
          and all(rate[p] - rate[fl.compose_wiring(p, m)] >= 0.05 for m in spec["miswirings"]))
    assert ok and g1["pass"], (f, "G1", rate)
    assert pf["agree"] / pf["total"] >= 0.97 and pf["pass"], (f, "preflight", pf)
print(f"verdict KILLED; all {n} shipped verdict fields reproduced; G1 and preflight recomputed from counts")
PYEOF

echo "3/3 mechanism decomposition"
"$PY" code/mechanism/analyze_mechanism.py "$WORK/run2" "$WORK/mech" > "$WORK/mech.log"
"$PY" - "$WORK/mech/mechanism.json" results/mechanism/mechanism.json <<'PYEOF'
import json, math, sys
def same(a, b, path="$"):
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            same(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and not (math.isnan(a) and math.isnan(b)):
        assert abs(a - b) < 1e-9, (path, a, b)
    else:
        assert a == b or (isinstance(a, float) and math.isnan(a) and math.isnan(b)), (path, a, b)
same(*(json.load(open(p)) for p in sys.argv[1:3]))
print("decomposition identical to results/mechanism/mechanism.json")
PYEOF
echo "ALL CHECKS PASSED"
