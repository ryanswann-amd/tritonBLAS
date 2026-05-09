# Routing Oracle — Negative Results & Falsifications

This page records candidate alias-stack slots that were **enumerated and
measured** but **rejected** at the productionization gate. Documenting
negative results prevents re-investigation and keeps the alias-stack lean
(every slot adds a per-call membership-test cost).

## Admission gates (current policy)

A new alias-stack `frozenset` slot is admitted only if, on its claimed
cohort cells (paired n=30 hot HIP-graph timing, MI300X gfx942):

- **Gate A**: gmean speedup ≥ 1.00× hipBLASLt
- **Gate B**: gmean speedup ≥ 1.05× current oracle (without the new slot)

Slots that route *out* to hipBLASLt are admitted only when the upstream
engine genuinely wins on the cohort; slots that route *in* to a Triton
tile config are admitted only when that tile genuinely beats both
references.

## Rejected: M=N=2048 K-COMPLEMENT JIT-tile override

**Cohort**: M=N=2048, K ∈ {2048, 4096, 8192, 16384}, dtype ∈ {bf16, fp16}
(8 cells). The cohort already routes 8/8 cells **out** to hipBLASLt via
existing slots (P12 square_mid, K971 LDS-BC, P26 N=2048 K-COMPL).

**Hypothesis**: One of the top-3 JIT-feasible tile configs from the
upstream tile-search enumeration could beat hipBLASLt on this cohort
when productionized as an alias-stack `frozenset` slot.

**Result**: **REJECTED.** Paired n=30 hot HIP-graph re-measurement
against the live 20-slot oracle and hipBLASLt baseline:

| Candidate (BK / NS / kpack)       | gmean vs hipBLASLt | Gate A | gmean vs oracle | Gate B |
|-----------------------------------|-------------------:|:------:|----------------:|:------:|
| BK=128, kpack=2                   | 0.870×             | FAIL   | 0.882×          | FAIL   |
| BK=128, kpack=default             | 0.852×             | FAIL   | 0.873×          | FAIL   |
| NS=3,   kpack=2                   | 0.845×             | FAIL   | 0.863×          | FAIL   |

0/8 cells admissible at any candidate. Best-of-best across 24
(candidate × cell) combinations is 1.083× — the best JIT config is
still 8.3% slower than hipBLASLt on its single best cell, and 27–34%
slower on the K=16384 cells.

**PMC root cause** (3-pass rocprofv2; gmean across 8 cells, BK=128 vs
hipBLASLt):

| Counter axis            | Ratio (TB / HBL) | Reading                                |
|-------------------------|-----------------:|----------------------------------------|
| MFMA-busy / cu-cyc      | 1.405×           | TB attains ~70% of HBL MFMA throughput |
| HBM-bytes / cu-cyc      | 0.879×           | TB advantage; not the bottleneck       |
| Active-CU%              | 1.139×           | TB slightly under-fills                |
| LDS_WAIT / wave         | **4.748×**       | TB blocks per wave 4–5× more on LDS    |
| GRBM runtime gap        | 1.420×           | TB is 42% slower wall-clock            |

`LDS_WAIT/wave` scales linearly with K (8.3 k cyc at K=2048 → 65.8 k
cyc at K=16384) while hipBLASLt stays flat near 2 k cyc/wave — the
`persistent_matmul` LDS-bank-conflict signature. Larger `BK` amortizes
per-wave MFMA overhead but doubles per-wave LDS work, so the
SCHEDULER-LDS budget caps the achievable speedup at ~0.87× on this
cohort. hipBLASLt's K-adaptive 304-wg variant (selected at K ≥ 4096)
has no counterpart in `persistent_matmul`'s fixed-grid family.

**Action**: No 21st alias-stack slot was added. The 20-slot routing
oracle is the correct production state for this cohort.
