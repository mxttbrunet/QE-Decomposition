#!/usr/bin/env python3
"""
mExps/merge_ablation.py
=======================
Controlled ablation that isolates the effect of MERGE's |K|>3 supernode-merge
step.  Both arms share an IDENTICAL decomposition pipeline so that any
difference in the result is attributable SOLELY to the merge.

Shared pipeline (both arms)
---------------------------
  * minimum-node-cut split; eliminate the smallest fragment V2.
  * EXACT ILP reweight: the boundary-fixed subproblem MaxCut is solved exactly
    by Gurobi for ALL 2^|K| cut-node colorings (mt.solvePartial), then a
    complete-graph fit with a NONNEGATIVE slack/"error" term e_i >= 0,
    minimizing sum(e_i).  Because e_i >= 0 the surrogate underestimates every
    assignment -> always a valid lower bound (overshoot-safe).
  * stop at |V| <= M; final reduced graph solved EXACTLY by brute force.

This is exactly the "ILP + error-term reweight using Gurobi" recipe, used by
BOTH arms.

The ONLY difference
-------------------
  NO-MERGE : when |K|>3 the |K|-node complete graph is reweighted directly.
             The pairwise surrogate cannot represent >2-way interactions, so the
             nonnegative slack underestimates -> valid but lossy.
  MERGE    : when |K|>3, QAOA-correlated cut nodes are first merged into
             supernodes (|K| -> <=3), THEN reweighted, so the pairwise surrogate
             is exact again -- unless the correlation-based merge groups nodes
             that should actually differ (its own error source).

Theory recap (why |K|<=3 is the boundary):
  The fragment value is flip-invariant, f(x)=f(~x), so it has 2^(|K|-1)
  distinct values; the surrogate has 1 + C(|K|,2) parameters.
      |K|=2 -> 2 values, 2 params  (exact)
      |K|=3 -> 4 values, 4 params  (exact, full-rank XOR basis)
      |K|=4 -> 8 values, 7 params  (NOT exact -> slack)
"""

import os
import sys
import time
import datetime
import io
import contextlib
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import numpy as np
import networkx as nx
import gurobipy as gp
from gurobipy import GRB

QAOA_DIR = "/home/mxttbrunet/QAOA-Graph-Decomp"
MERGE_DIR = "/home/mxttbrunet/QE-Decomposition"
sys.path.insert(0, QAOA_DIR)
sys.path.insert(0, MERGE_DIR)

import MERGE_TEMP1 as mt          # getInduced, mergeAndUpdate2, bruteMaxCut, solvePartial, M, tau
mt.draw = lambda *a, **k: None    # disable plotting
import COMP_DECOMP as cd          # cd.solver = exact optimal MaxCut

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MERGE_ABLATION.txt")

N_GRAPHS = 30
N_NODES = 14
DEGREE = 3


def build_graphs():
    gs = []
    for i in range(N_GRAPHS):
        G = nx.random_regular_graph(DEGREE, N_NODES, seed=i)
        for u, v in G.edges():
            G[u][v]['weight'] = 1
        gs.append(G)
    return gs


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


# ---------------------------------------------------------------------------
# Exact-ILP + nonnegative-slack reweight (shared by both arms).
# Returns (J dict, total slack = sum e_i).
# ---------------------------------------------------------------------------
def reweightFull(sub, K, V2):
    K = list(K)
    sub = sub.copy()
    Kset = set(K)
    sub.remove_edges_from([(u, v) for u, v in sub.edges() if u in Kset and v in Kset])

    kM = len(K)
    assigns = [[int(c) for c in s] for s in mt.genPerms(kM)]
    RHS = [quiet(mt.solvePartial, a, sub, K) for a in assigns]
    constant = RHS[0]
    if kM == 1:
        return {'seaHat': min(RHS)}, 0.0

    shifted = [r - constant for r in RHS]
    idxPairs = list(combinations(range(kM), 2))
    C = [[a[i] ^ a[j] for (i, j) in idxPairs] for a in assigns]

    model = gp.Model("reweightFull")
    model.setParam("OutputFlag", 0)
    e = [model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="e%d" % i) for i in range(len(assigns))]
    w = [model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="w%d" % j) for j in range(len(idxPairs))]
    model.setObjective(gp.quicksum(e), GRB.MINIMIZE)
    for i in range(len(assigns)):
        model.addConstr(e[i] + gp.quicksum(C[i][j] * w[j] for j in range(len(idxPairs))) == shifted[i])
    model.optimize()

    slack = float(model.ObjVal)
    edgePairs = list(combinations(K, 2))
    res = {'seaHat': constant}
    for j, (u, v) in enumerate(edgePairs):
        res[(u, v)] = w[j].X
    return res, slack


# ---------------------------------------------------------------------------
# Unified decomposition; use_merge is the single toggle under study.
# ---------------------------------------------------------------------------
def run_decompo(g, limit, use_merge):
    gNum = -1
    cg = g.copy()
    cIsh = 0.0
    fixTable = {}
    log = []
    rnd = 1
    total_slack = 0.0
    n_big = 0       # rounds with |K|>3
    n_merged = 0    # rounds where a merge actually collapsed >=1 supernode
    max_k = 0
    t0 = time.perf_counter()
    while len(cg.nodes()) > limit:
        try:
            V2andK, K, V1, V2 = quiet(mt.getInduced, cg)
        except Exception as ex:
            log.append((rnd, None, "stop: no node cut (%s)" % type(ex).__name__))
            break
        rawK = list(K)
        max_k = max(max_k, len(rawK))
        if len(rawK) > 10:
            log.append((rnd, rawK, "stop: |K|=%d too large for Gurobi size-limited license" % len(rawK)))
            break
        note = ""
        if len(K) > 3:
            n_big += 1
            if use_merge:
                mergedN, cg, K, _iw = quiet(mt.mergeAndUpdate2, cg, V2andK, K, V2, gNum)
                gNum -= len(mergedN)
                quiet(mt.tableUpdate, mergedN, fixTable)
                K = list(K) + list(mergedN.keys())          # supernode kept in cut set
                V2andK = cg.subgraph(list(K) + list(V2))
                if mergedN:
                    n_merged += 1
                note = "  (|K|>3; merged supernodes %s -> reweight |K|=%d)" % (list(mergedN.keys()), len(K))
            else:
                note = "  (|K|>3; NO merge -> lossy pairwise reweight)"

        J, slack = reweightFull(V2andK, K, V2)
        total_slack += slack
        cg.remove_nodes_from(V2)
        for edge, val in list(J.items()):
            if edge == 'seaHat':
                cIsh += val
            elif (edge[0], edge[1]) in cg.edges():
                cg[edge[0]][edge[1]]['weight'] += val
            else:
                cg.add_edge(edge[0], edge[1])
                cg[edge[0]][edge[1]]['weight'] = val

        log.append((rnd, rawK, "removed |V2|=%d, |K|=%d%s" % (len(V2), len(rawK), note)))
        rnd += 1

    bestCut, _ = quiet(mt.bruteMaxCut, cg)
    elapsed = time.perf_counter() - t0
    return {
        "cut": bestCut + cIsh,
        "final_nodes": len(cg.nodes()),
        "n_big": n_big,
        "n_merged": n_merged,
        "max_k": max_k,
        "total_slack": total_slack,
        "time": elapsed,
        "log": log,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_report(rows):
    nm = np.array([r["nomerge"]["cut"] for r in rows])
    mm = np.array([r["merge"]["cut"] for r in rows])
    opt = np.array([r["opt"] for r in rows])
    nm_r, mm_r = nm / opt, mm / opt
    nm_e, mm_e = np.abs(nm - opt), np.abs(mm - opt)

    L = []
    L.append("=" * 84)
    L.append("MERGE ABLATION  --  isolating the |K|>3 supernode-merge step")
    L.append("=" * 84)
    L.append("Generated: %s" % datetime.datetime.now().isoformat(timespec="seconds"))
    L.append("Benchmark: %d random %d-regular graphs on %d nodes (seed = graph index)"
             % (N_GRAPHS, DEGREE, N_NODES))
    L.append("")
    L.append("Both arms: identical pipeline -- min-node-cut split, eliminate smallest")
    L.append("fragment, EXACT ILP reweight (all 2^|K| colorings, Gurobi) + nonnegative")
    L.append("slack/error LP, stop at |V|<=M=%d, EXACT brute-force final solve." % mt.M)
    L.append("Only difference: NO-MERGE reweights |K|>3 cliques directly (lossy);")
    L.append("MERGE collapses QAOA-correlated cut nodes to |K|<=3 first (exact-again).")
    L.append("Merge arm uses a shot-based QAOA simulator -> mildly stochastic.")
    L.append("")
    L.append("-" * 84)
    L.append("OVERALL SUMMARY (%d graphs)" % len(rows))
    L.append("-" * 84)
    L.append("%-30s %14s %14s" % ("metric", "NO-MERGE", "MERGE"))
    L.append("%-30s %14.6f %14.6f" % ("mean approx ratio", nm_r.mean(), mm_r.mean()))
    L.append("%-30s %14.6f %14.6f" % ("min approx ratio", nm_r.min(), mm_r.min()))
    L.append("%-30s %14.6f %14.6f" % ("max approx ratio", nm_r.max(), mm_r.max()))
    L.append("%-30s %14.6f %14.6f" % ("mean |error| from optimal", nm_e.mean(), mm_e.mean()))
    L.append("%-30s %14d %14d" % ("exact-optimal hits", int((nm_e <= 1e-6).sum()), int((mm_e <= 1e-6).sum())))
    L.append("%-30s %14d %14d" % ("overshoots (ratio>1)", int((nm_r > 1 + 1e-6).sum()), int((mm_r > 1 + 1e-6).sum())))
    L.append("%-30s %14.4f %14.4f" % ("total decomp time (s)",
             sum(r["nomerge"]["time"] for r in rows), sum(r["merge"]["time"] for r in rows)))
    L.append("")
    same = int(np.sum(np.abs(nm - mm) <= 1e-6))
    merge_better = int(np.sum(mm_e < nm_e - 1e-6))
    nomerge_better = int(np.sum(nm_e < mm_e - 1e-6))
    L.append("Head-to-head (accuracy vs exact optimum):")
    L.append("  identical result            : %d graphs" % same)
    L.append("  MERGE strictly closer       : %d graphs" % merge_better)
    L.append("  NO-MERGE strictly closer    : %d graphs" % nomerge_better)
    L.append("")
    L.append("VALIDATION: on graphs with zero |K|>3 rounds the two arms MUST be")
    L.append("identical (the merge never fires). Divergence appears only where |K|>3.")
    L.append("")
    L.append("-" * 84)
    L.append("PER-GRAPH COMPARISON")
    L.append("-" * 84)
    L.append("%5s %7s %10s %10s %8s %8s %8s %8s %s" %
             ("graph", "optimal", "NOMERGE", "MERGE", "NM_ratio", "MG_ratio",
              "|K|>3", "merged", "verdict"))
    for r in rows:
        a, b = r["nomerge"], r["merge"]
        ec1, ec2 = abs(a["cut"] - r["opt"]), abs(b["cut"] - r["opt"])
        if abs(a["cut"] - b["cut"]) <= 1e-6:
            v = "same"
        elif ec2 < ec1 - 1e-6:
            v = "MERGE+"
        else:
            v = "NOMERGE+"
        L.append("%5d %7.1f %10.4f %10.4f %8.4f %8.4f %8d %8d %s" %
                 (r["idx"], r["opt"], a["cut"], b["cut"], a["cut"] / r["opt"],
                  b["cut"] / r["opt"], a["n_big"], b["n_merged"], v))
    L.append("")
    L.append("-" * 84)
    L.append("DIAGNOSTICS  (NO-MERGE total slack lost = sum of e_i across all rounds)")
    L.append("-" * 84)
    L.append("%5s %12s %12s %10s %10s" %
             ("graph", "NM_slack", "NM_maxK", "NM_finalV", "MG_finalV"))
    for r in rows:
        a, b = r["nomerge"], r["merge"]
        L.append("%5d %12.4f %12d %10d %10d" %
                 (r["idx"], a["total_slack"], a["max_k"], a["final_nodes"], b["final_nodes"]))
    L.append("")
    L.append("=" * 84)
    L.append("PER-GRAPH ROUND LOGS")
    L.append("=" * 84)
    for r in rows:
        L.append("")
        L.append("GRAPH %d  (optimal=%.0f)" % (r["idx"], r["opt"]))
        L.append("  NO-MERGE  -> cut=%.4f  ratio=%.4f" % (r["nomerge"]["cut"], r["nomerge"]["cut"] / r["opt"]))
        for rd, K, outcome in r["nomerge"]["log"]:
            ks = "n/a" if K is None else "{" + ", ".join(map(str, K)) + "}"
            L.append("      round %2d : K=%s -> %s" % (rd, ks, outcome))
        L.append("  MERGE     -> cut=%.4f  ratio=%.4f" % (r["merge"]["cut"], r["merge"]["cut"] / r["opt"]))
        for rd, K, outcome in r["merge"]["log"]:
            ks = "n/a" if K is None else "{" + ", ".join(map(str, K)) + "}"
            L.append("      round %2d : K=%s -> %s" % (rd, ks, outcome))
    L.append("")
    L.append("=" * 84)
    with open(OUT_PATH, "w") as f:
        f.write("\n".join(L) + "\n")
    print("Wrote", OUT_PATH)


def main():
    graphs = build_graphs()
    rows = []
    for i, G in enumerate(graphs):
        opt = quiet(cd.solver, G)
        np.random.seed(0)
        nomerge = run_decompo(G.copy(), mt.M, use_merge=False)
        np.random.seed(0)
        merge = run_decompo(G.copy(), mt.M, use_merge=True)
        rows.append({"idx": i, "opt": opt, "nomerge": nomerge, "merge": merge})
        print("graph %2d opt=%.1f | NO-MERGE %.4f (r=%.4f, |K|>3=%d) | MERGE %.4f (r=%.4f, merged=%d)"
              % (i, opt, nomerge["cut"], nomerge["cut"] / opt, nomerge["n_big"],
                 merge["cut"], merge["cut"] / opt, merge["n_merged"]))
    write_report(rows)


if __name__ == "__main__":
    main()
