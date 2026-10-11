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
cp -R data/run2 "$WORK/run2"

echo "2/3 verdict"
"$PY" code/probe/verdict.py "$WORK/run2" > "$WORK/verdict.log"
"$PY" - "$WORK/run2/verdict.json" data/run2/verdict.json <<'PYEOF'
import json, sys
new, old = (json.load(open(p)) for p in sys.argv[1:3])
assert new["verdict"] == old["verdict"] == "KILLED", (new["verdict"], old["verdict"])
assert not new.get("deviations"), new.get("deviations")
for f in ("qwen", "gemma"):
    a, b = new["families"][f], old["families"][f]
    for k in ("partial_spearman", "raw_spearman", "fires"):
        assert a["F1"][k] == b["F1"][k], (f, "F1", k, a["F1"][k], b["F1"][k])
    for k in ("median", "fires", "n_langs"):
        assert a["F2p"][k] == b["F2p"][k], (f, "F2p", k, a["F2p"][k], b["F2p"][k])
    for k in ("ci95",):
        assert max(abs(x - y) for x, y in zip(a["F1"][k], b["F1"][k])) < 1e-12, (f, "F1 CI")
    assert a["G1"]["pass"] and a["G2"]["pass"] and a["G3"]["pass"], f
print("verdict KILLED; F1 and F2' statistics identical to the shipped verdict.json")
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
