# mExps — isolating the effect of MERGE's node-merge step

These experiments answer one question: **how much of MERGE FINAL's advantage over
QAOA-Decomposition comes from the `|K|>3` supernode-merge step itself**, as opposed
to the other differences (exact ILP vs p=1 QAOA reweighting, exact vs approximate
final solve)?

The original comparison (`../FINAL_ANAL.txt`, produced by
`../../QE-Decomposition/MERGE_0701.py`) conflated several differences: it ran
QAOA-Decomposition on its **p=1 QAOA** reduction path (`COMP_DECOMP.ReductionG`),
while MERGE used exact ILP reweighting *and* the merge step. So the headline gap
(0.929 vs 0.995) measured "QAOA vs exact" mixed with "merge vs no-merge."

## `merge_ablation.py` — the controlled experiment

Both arms run an **identical** pipeline:

- minimum-node-cut split; eliminate the smallest fragment `V2`;
- **exact ILP reweight**: boundary-fixed subproblem MaxCut solved exactly by Gurobi
  for all `2^|K|` cut-node colorings, then a complete-graph fit with a
  **nonnegative slack/error term** `e_i ≥ 0`, minimizing `Σ e_i` (overshoot-safe);
- stop at `|V| ≤ M`; final reduced graph solved **exactly by brute force**.

The **only** toggle is `use_merge`:

- **NO-MERGE** (QAOA-Decomp reweight philosophy): when `|K|>3`, reweight the
  `|K|`-node clique directly. The pairwise surrogate cannot represent >2-way
  interactions, so the nonnegative slack underestimates → valid but lossy.
- **MERGE**: when `|K|>3`, first merge QAOA-correlated cut nodes into supernodes
  (`|K| → ≤3`), then reweight → pairwise-exact again (unless the correlation-based
  merge groups nodes that should differ — its own error source).

Why `|K|≤3` is the boundary: the fragment value is flip-invariant `f(x)=f(¬x)`, so it
has `2^(|K|-1)` distinct values, while the surrogate has `1 + C(|K|,2)` parameters.
`|K|=2,3` → params ≥ values (exact); `|K|≥4` → params < values (slack appears).

Run: `python3 merge_ablation.py` → writes `MERGE_ABLATION.txt`.
(The MERGE arm uses a shot-based QAOA simulator, so it is mildly stochastic.)

### Headline result (30 seeded 3-regular 14-node graphs)

| metric | NO-MERGE | MERGE |
|---|---|---|
| mean approx ratio | 0.9894 | 0.9946 |
| exact-optimal hits | 24/30 | 27/30 |
| overshoots | 0 | 0 |

- **14 graphs have zero `|K|>3` rounds → the two arms are byte-identical and both
  optimal.** This validates that the pipeline really is identical and the merge is
  the sole lever.
- Head-to-head: MERGE strictly better on **4** graphs (20, 22, 24, 29 — recovers the
  slack lost by direct `|K|>3` reweight), NO-MERGE strictly better on **1** graph
  (1 — the correlation merge collapsed nodes that should have differed), identical on
  the other 25.
- **Takeaway:** most of MERGE FINAL's edge over the original QAOA-Decomp (0.929 →
  ~0.99) came from using exact ILP reweighting + an exact final solve. The merge step
  adds a smaller, *net-positive-but-not-free* increment on top (≈ +0.005 mean,
  +3 exact hits), and can occasionally hurt.

## `trace_graph3.py` — single-graph mechanism demo

Walks one graph (#3) through both the exact and p=1 QAOA subproblem solvers (using
`COMP_DECOMP`'s own functions) to show, concretely, *why* p=1 QAOA reweighting loses
information (it returns the same value for every boundary coloring), and that the
exact ILP + slack reweight is lossless on `|K|≤3` rounds.

Run: `python3 trace_graph3.py`.
