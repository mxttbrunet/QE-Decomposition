#!/usr/bin/env python3
"""
MERGE_PS_100/merge_ps.py
========================
Threshold sweep of the MERGE node-merge step, using the SAME controlled ablation
as ../mExps/merge_ablation.py, across merge thresholds tau in {0.1, 0.2, 0.3, 0.4}.

Benchmark : 10 random 4-regular graphs on 20 nodes (seed = graph index).
(4-regular -> min node cut ~4, so |K|>3 rounds -- where the merge fires -- are
the common case, making this a strong stress test of the merge step.)

For every graph both arms run an IDENTICAL pipeline (min-node-cut split, exact
ILP reweight with nonnegative slack/error term, exact brute-force final solve);
the ONLY difference is whether |K|>3 rounds first merge QAOA-correlated cut nodes
into supernodes.  The merge threshold tau decides which cut-node pairs are
"correlated enough" to merge:  pair merged iff  <Z_i Z_j> >= tau.

  * NO-MERGE is tau-independent -> computed once, shared across all four reports.
  * MERGE(tau) is recomputed for each tau.

Outputs (this directory):
  COMP_T0.1.txt  COMP_T0.2.txt  COMP_T0.3.txt  COMP_T0.4.txt   (NO-MERGE vs MERGE(tau))
  COMP_BY_T.txt                                                 (all thresholds together)
"""

import os
import sys
import datetime
import numpy as np
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mExps"))   # reuse the ablation machinery
sys.path.insert(0, "/home/mxttbrunet/QAOA-Graph-Decomp")
sys.path.insert(0, "/home/mxttbrunet/QE-Decomposition")

import merge_ablation as ab          # run_decompo, reweightFull, quiet, mt, cd
mt = ab.mt
cd = ab.cd

N_GRAPHS = 10
N_NODES = 20
DEGREE = 4
TAUS = [0.1, 0.2, 0.3, 0.4]


def build_graphs():
    gs = []
    for i in range(N_GRAPHS):
        G = nx.random_regular_graph(DEGREE, N_NODES, seed=i)
        for u, v in G.edges():
            G[u][v]['weight'] = 1
        gs.append(G)
    return gs


def tag(tau):
    return ("%.1f" % tau)


# ---------------------------------------------------------------------------
# Per-threshold report: NO-MERGE vs MERGE(tau)
# ---------------------------------------------------------------------------
def write_threshold_report(tau, opt, nm, mg, path):
    opt = np.array(opt, float)
    nmc = np.array([r["cut"] for r in nm], float)
    mgc = np.array([r["cut"] for r in mg], float)
    nm_r, mg_r = nmc / opt, mgc / opt
    nm_e, mg_e = np.abs(nmc - opt), np.abs(mgc - opt)

    same = int(np.sum(np.abs(nmc - mgc) <= 1e-6))
    merge_better = int(np.sum(mg_e < nm_e - 1e-6))
    nomerge_better = int(np.sum(nm_e < mg_e - 1e-6))

    L = []
    L.append("=" * 84)
    L.append("MERGE ABLATION  --  threshold tau = %s" % tag(tau))
    L.append("NO-MERGE  vs  MERGE(tau=%s)" % tag(tau))
    L.append("=" * 84)
    L.append("Generated: %s" % datetime.datetime.now().isoformat(timespec="seconds"))
    L.append("Benchmark: %d random %d-regular graphs on %d nodes (seed = graph index)"
             % (N_GRAPHS, DEGREE, N_NODES))
    L.append("")
    L.append("Both arms: identical pipeline (min-node-cut split, exact ILP reweight +")
    L.append("nonnegative slack LP, stop at |V|<=M=%d, exact brute-force final solve)." % mt.M)
    L.append("Only difference: MERGE collapses cut-node pairs with <Z_iZ_j> >= %s into" % tag(tau))
    L.append("supernodes when |K|>3, then reweights; NO-MERGE reweights the clique directly.")
    L.append("Merge arm uses a shot-based QAOA simulator -> mildly stochastic.")
    L.append("")
    L.append("-" * 84)
    L.append("OVERALL SUMMARY (%d graphs)" % N_GRAPHS)
    L.append("-" * 84)
    L.append("%-30s %14s %14s" % ("metric", "NO-MERGE", "MERGE(%s)" % tag(tau)))
    L.append("%-30s %14.6f %14.6f" % ("mean approx ratio", nm_r.mean(), mg_r.mean()))
    L.append("%-30s %14.6f %14.6f" % ("min approx ratio", nm_r.min(), mg_r.min()))
    L.append("%-30s %14.6f %14.6f" % ("max approx ratio", nm_r.max(), mg_r.max()))
    L.append("%-30s %14.6f %14.6f" % ("mean |error| from optimal", nm_e.mean(), mg_e.mean()))
    L.append("%-30s %14d %14d" % ("exact-optimal hits", int((nm_e <= 1e-6).sum()), int((mg_e <= 1e-6).sum())))
    L.append("%-30s %14d %14d" % ("overshoots (ratio>1)", int((nm_r > 1 + 1e-6).sum()), int((mg_r > 1 + 1e-6).sum())))
    L.append("%-30s %14.4f %14.4f" % ("total decomp time (s)",
             sum(r["time"] for r in nm), sum(r["time"] for r in mg)))
    L.append("")
    L.append("Head-to-head (accuracy vs exact optimum):")
    L.append("  identical result            : %d graphs" % same)
    L.append("  MERGE strictly closer       : %d graphs" % merge_better)
    L.append("  NO-MERGE strictly closer    : %d graphs" % nomerge_better)
    L.append("")
    L.append("-" * 84)
    L.append("PER-GRAPH COMPARISON")
    L.append("-" * 84)
    L.append("%5s %7s %10s %10s %9s %9s %7s %7s %s" %
             ("graph", "optimal", "NOMERGE", "MERGE", "NM_ratio", "MG_ratio", "|K|>3", "merged", "verdict"))
    for i in range(N_GRAPHS):
        ec1, ec2 = nm_e[i], mg_e[i]
        if abs(nmc[i] - mgc[i]) <= 1e-6:
            v = "same"
        elif ec2 < ec1 - 1e-6:
            v = "MERGE+"
        else:
            v = "NOMERGE+"
        L.append("%5d %7.1f %10.4f %10.4f %9.4f %9.4f %7d %7d %s" %
                 (i, opt[i], nmc[i], mgc[i], nm_r[i], mg_r[i],
                  nm[i]["n_big"], mg[i]["n_merged"], v))
    L.append("")
    L.append("=" * 84)
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("Wrote", path)
    return {
        "tau": tau, "mean_r": mg_r.mean(), "min_r": mg_r.min(),
        "mean_e": mg_e.mean(), "exact": int((mg_e <= 1e-6).sum()),
        "over": int((mg_r > 1 + 1e-6).sum()), "same": same,
        "merge_better": merge_better, "nomerge_better": nomerge_better,
        "time": sum(r["time"] for r in mg), "cuts": mgc,
    }


# ---------------------------------------------------------------------------
# Combined report across all thresholds
# ---------------------------------------------------------------------------
def write_combined(opt, nm, per_tau, path):
    opt = np.array(opt, float)
    nmc = np.array([r["cut"] for r in nm], float)
    nm_r = nmc / opt
    nm_e = np.abs(nmc - opt)

    L = []
    L.append("=" * 96)
    L.append("COMPARISON BY MERGE THRESHOLD  (COMP_BY_T)")
    L.append("=" * 96)
    L.append("Generated: %s" % datetime.datetime.now().isoformat(timespec="seconds"))
    L.append("Benchmark: %d random %d-regular graphs on %d nodes (seed = graph index)"
             % (N_GRAPHS, DEGREE, N_NODES))
    L.append("")
    L.append("Controlled ablation: both arms share an identical exact-ILP + nonnegative-")
    L.append("slack reweight pipeline with an exact brute-force final solve; the only")
    L.append("difference is the |K|>3 supernode-merge.  The merge threshold tau sets the")
    L.append("ZZ-correlation cutoff for merging cut-node pairs.  NO-MERGE is the common,")
    L.append("tau-independent baseline; each MERGE(tau) is compared against it.")
    L.append("")
    L.append("-" * 96)
    L.append("BASELINE  NO-MERGE  (tau-independent)")
    L.append("-" * 96)
    L.append("  mean approx ratio        : %.6f" % nm_r.mean())
    L.append("  min approx ratio         : %.6f" % nm_r.min())
    L.append("  mean |error| from optimal: %.6f" % nm_e.mean())
    L.append("  exact-optimal hits       : %d / %d" % (int((nm_e <= 1e-6).sum()), N_GRAPHS))
    L.append("  overshoots (ratio>1)     : %d" % int((nm_r > 1 + 1e-6).sum()))
    L.append("  total decomp time (s)    : %.4f" % sum(r["time"] for r in nm))
    L.append("")
    L.append("-" * 96)
    L.append("MERGE(tau)  SUMMARY  vs the common NO-MERGE baseline")
    L.append("-" * 96)
    L.append("%6s %9s %9s %10s %7s %6s %7s %12s %12s %9s" %
             ("tau", "mean_r", "min_r", "mean|err|", "exact", "over",
              "same", "MERGE_wins", "NOMERGE_wins", "time_s"))
    L.append("%6s %9.5f %9.5f %10.5f %7d %6d %7s %12s %12s %9.2f" %
             ("none", nm_r.mean(), nm_r.min(), nm_e.mean(),
              int((nm_e <= 1e-6).sum()), int((nm_r > 1 + 1e-6).sum()),
              "-", "-", "-", sum(r["time"] for r in nm)))
    for s in per_tau:
        L.append("%6s %9.5f %9.5f %10.5f %7d %6d %7d %12d %12d %9.2f" %
                 (tag(s["tau"]), s["mean_r"], s["min_r"], s["mean_e"], s["exact"],
                  s["over"], s["same"], s["merge_better"], s["nomerge_better"], s["time"]))
    L.append("")
    L.append("(exact = #graphs hitting the exact optimum; over = #overshoots;")
    L.append(" same/MERGE_wins/NOMERGE_wins = head-to-head vs NO-MERGE on |error|.)")
    L.append("")
    best = max(per_tau, key=lambda s: (s["exact"], s["mean_r"]))
    L.append("Best threshold by exact hits then mean ratio: tau = %s "
             "(mean_r=%.5f, exact=%d/%d)."
             % (tag(best["tau"]), best["mean_r"], best["exact"], N_GRAPHS))
    L.append("")
    L.append("-" * 96)
    L.append("PER-GRAPH MAXCUT  (optimal | NO-MERGE | MERGE at each tau)")
    L.append("-" * 96)
    hdr = "%5s %8s %10s" % ("graph", "optimal", "NOMERGE")
    for s in per_tau:
        hdr += " %10s" % ("MERGE@" + tag(s["tau"]))
    L.append(hdr)
    for i in range(N_GRAPHS):
        row = "%5d %8.1f %10.4f" % (i, opt[i], nmc[i])
        for s in per_tau:
            row += " %10.4f" % s["cuts"][i]
        L.append(row)
    L.append("")
    L.append("-" * 96)
    L.append("PER-GRAPH APPROX RATIO  (NO-MERGE | MERGE at each tau)")
    L.append("-" * 96)
    hdr = "%5s %8s %10s" % ("graph", "optimal", "NOMERGE")
    for s in per_tau:
        hdr += " %10s" % ("MERGE@" + tag(s["tau"]))
    L.append(hdr)
    for i in range(N_GRAPHS):
        row = "%5d %8.1f %10.4f" % (i, opt[i], nm_r[i])
        for s in per_tau:
            row += " %10.4f" % (s["cuts"][i] / opt[i])
        L.append(row)
    L.append("")
    L.append("=" * 96)
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("Wrote", path)


def main():
    graphs = build_graphs()

    # ---- exact optimal + tau-independent NO-MERGE baseline (once) ----
    opt = []
    nm = []
    for i, G in enumerate(graphs):
        opt.append(ab.quiet(cd.solver, G))
        np.random.seed(0)
        nm.append(ab.run_decompo(G.copy(), mt.M, use_merge=False))
        print("baseline graph %2d opt=%.1f NO-MERGE=%.4f (|K|>3=%d)"
              % (i, opt[i], nm[i]["cut"], nm[i]["n_big"]))

    # ---- MERGE(tau) for each threshold ----
    per_tau = []
    for tau in TAUS:
        mt.tau = tau
        mg = []
        for i, G in enumerate(graphs):
            np.random.seed(0)
            mg.append(ab.run_decompo(G.copy(), mt.M, use_merge=True))
            print("tau=%s graph %2d MERGE=%.4f (r=%.4f, merged=%d)"
                  % (tag(tau), i, mg[i]["cut"], mg[i]["cut"] / opt[i], mg[i]["n_merged"]))
        path = os.path.join(HERE, "COMP_T%s.txt" % tag(tau))
        per_tau.append(write_threshold_report(tau, opt, nm, mg, path))

    write_combined(opt, nm, per_tau, os.path.join(HERE, "COMP_BY_T.txt"))


if __name__ == "__main__":
    main()
