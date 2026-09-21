#!/usr/bin/env bash
# Drive the whole benchmark grid across the four backend environments.
#
# The backends cannot share a Python environment: NetQASM/SquidASM needs
# netqasm 1.x, Qoala needs netqasm 2.x (mutually incompatible), Aer wants a
# recent Qiskit, and CUNQA only exists inside its container with Slurm. So
# each backend is invoked through its own interpreter and every run is a
# fresh process, which is also what keeps peak-RSS and import timings clean.
#
# Every configuration appends one JSON record per repetition to
# results/raw.jsonl. Records are never lost: a backend that cannot run an
# app records why, and that is what the portability matrix is built from.
#
# Usage:
#   ./run_all.sh              # full grid
#   ./run_all.sh --quick      # a small smoke grid
#   BACKENDS="aer cunqa" ./run_all.sh
set -uo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${BENCH_DIR}/../.." && pwd)"
OUT="${BENCH_DIR}/results/raw.jsonl"
RUNNER="scripts/benchmark/run_benchmark.py"

CONDA_ENVS="${CONDA_ENVS:-$HOME/miniconda3/envs}"
PY_AER="${PY_AER:-$CONDA_ENVS/qiskit-1-3/bin/python}"
PY_QOALA="${PY_QOALA:-$CONDA_ENVS/qoala/bin/python}"
PY_NETQASM="${PY_NETQASM:-$CONDA_ENVS/squidasm/bin/python}"
DOCKER_IMAGE="${DOCKER_IMAGE:-netqmpi:cunqa}"

QUICK=0
[[ "${1:-}" == "--quick" ]] && QUICK=1

BACKENDS="${BACKENDS:-aer cunqa qoala netqasm}"
APPS="${APPS:-cascade ghz qft qft_telegate}"

if (( QUICK )); then
    RANKS_AER="2 3";        QUBITS_AER="1"
    RANKS_CUNQA="2 3";      QUBITS_CUNQA="1"
    RANKS_QOALA="2";        RANKS_NETQASM="2"
    REPS=1
else
    RANKS_AER="2 3 4 5 6";  QUBITS_AER="1 2 3"
    RANKS_CUNQA="2 3 4 5 6 7"; QUBITS_CUNQA="1"
    RANKS_QOALA="2 3 4";    RANKS_NETQASM="2 3"
    REPS=3
fi

mkdir -p "${BENCH_DIR}/results"
: > "$OUT"
echo "grid -> $OUT"

# ----------------------------------------------------------------------
# Aer — fast and noiseless, so it doubles as the correctness reference.
# ----------------------------------------------------------------------
run_aer() {
    [[ -x "$PY_AER" ]] || { echo "skip aer: no interpreter at $PY_AER"; return; }
    for app in $APPS; do
        for n in $RANKS_AER; do
            for q in $QUBITS_AER; do
                PYTHONPATH="$REPO_ROOT" "$PY_AER" "$REPO_ROOT/$RUNNER" \
                    --backend aer --app "$app" --ranks "$n" --qubits "$q" \
                    --shots 1024 --reps "$REPS" --memory \
                    --timeout 120 --out "$OUT"
            done
        done
    done
}

# ----------------------------------------------------------------------
# CUNQA — one container for the whole sweep, since starting it also starts
# Slurm. Each run raises and drops its own vQPUs so the setup phase measures
# the real cost of acquiring them.
#
# The sweep stays at one qubit per rank and grows the rank count instead: the
# default vQPU definition is narrow, and two data qubits plus the scratch slot
# plus the communication qubits already overflow it ("Not enough data qubits
# in the QPU for the circuit"). Widening it needs a vQPU definition file
# passed as the 'backend' setting, which is a property of the deployment
# rather than of NetQMPI.
# ----------------------------------------------------------------------
run_cunqa() {
    command -v docker >/dev/null || { echo "skip cunqa: no docker"; return; }
    local script="export PYTHONPATH=/work:\$PYTHONPATH; set -u"
    for app in $APPS; do
        for n in $RANKS_CUNQA; do
            for q in $QUBITS_CUNQA; do
                script+="
python3 $RUNNER --backend cunqa --app $app --ranks $n --qubits $q \
    --shots 1024 --reps $REPS --memory --cunqa-qraise \
    --timeout 300 --out /work/scripts/benchmark/results/raw.jsonl"
            done
        done
    done
    docker run --rm -v "$REPO_ROOT":/work -w /work "$DOCKER_IMAGE" \
        bash -lc "$script" 2>&1 | grep -Ev '^\[entrypoint\]|^$'
}

# ----------------------------------------------------------------------
# Qoala — NetSquid simulation, so shots are expensive; the grid stays small.
# ----------------------------------------------------------------------
run_qoala() {
    [[ -x "$PY_QOALA" ]] || { echo "skip qoala: no interpreter at $PY_QOALA"; return; }
    for app in $APPS; do
        for n in $RANKS_QOALA; do
            PYTHONPATH="$REPO_ROOT" "$PY_QOALA" "$REPO_ROOT/$RUNNER" \
                --backend qoala --app "$app" --ranks "$n" --qubits 1 \
                --shots 100 --reps "$REPS" --memory \
                --timeout 300 --out "$OUT"
        done
    done
}

# ----------------------------------------------------------------------
# NetQASM/SquidASM — by far the slowest path: a handful of shots on two or
# three ranks is all that fits in a sensible wall-clock budget.
# ----------------------------------------------------------------------
run_netqasm() {
    [[ -x "$PY_NETQASM" ]] || { echo "skip netqasm: no interpreter at $PY_NETQASM"; return; }
    for app in $APPS; do
        for n in $RANKS_NETQASM; do
            PYTHONPATH="$REPO_ROOT" "$PY_NETQASM" "$REPO_ROOT/$RUNNER" \
                --backend netqasm --app "$app" --ranks "$n" --qubits 1 \
                --shots 10 --reps 1 --memory \
                --timeout 600 --out "$OUT"
        done
    done
}

for backend in $BACKENDS; do
    echo "=== $backend ==="
    "run_${backend}"
done

echo
echo "records: $(wc -l < "$OUT")"
echo "next: python scripts/benchmark/analyze.py"
