#!/usr/bin/env python3
"""
MERGE_PS_5REG/merge_ps_repeat.py
================================
Denoised re-run of the degree-5 threshold sweep.  The shot-based QAOA merge makes
single-shot per-graph results high-variance at degree 5, so here each MERGE(tau)
result is run REPEATS times (different np seed per repeat) and aggregated.

Benchmark : 10 random 5-regular graphs on 20 nodes (seed = graph index).
NO-MERGE has no quantum step -> deterministic -> run once.

Outputs:
  COMP_T0.1_avg.txt ... COMP_T0.4_avg.txt   (per-tau, per-graph repeat statistics)
  COMP_BY_T_REPEAT.txt                       (denoised summary across thresholds)
"""

import os
import sys
import datetime
import numpy as np
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mExps"))
sys.path.insert(0, "/home/mxttbrunet/QAOA-Graph-Decomp")
sys.path.insert(0, "/home/mxttbrunet/QE-Decomposition")

import merge_ablation as ab
mt = ab.mt
cd = ab.cd

N_GRAPHS = 10
N_NODES = 20
DEGREE = 5
TAUS = [0.1, 0.2, 0.3, 0.4]
REPEATS = 5


def build_graphs():
    gs = []
    for i in range(N_GRAPHS):
        G = nx.random_regular_graph(DEGREE, N_NODES, seed=i)
        for u, v in G.edges():
            G[u][v]['weight'] = 1
        gs.append(G)
    return gs


def tag(tau):
    return "%.1f" % tau


def per_graph_stats(cuts, opt):
    a = np.array(cuts, float)
    return {
        "mean": a.mean(), "std": a.std(), "min": a.min(), "max": a.max(),
        "mean_r": a.mean() / opt, "best_r": a.max() / opt, "worst_r": a.min() / opt,
        "exact_any": bool(np.any(np.abs(a - opt) <= 1e-6)),
        "exact_count": int(np.sum(np.abs(a - opt) <= 1e-6)),
    }


def write_tau_avg(tau, opt, nm, stats, path):
    L = []
    L.append("=" * 92)
    L.append("DEGREE-5 DENOISED  --  threshold tau = %s   (%d repeats per graph)" % (tag(tau), REPEATS))
    L.append("=" * 92)
    L.append("Generated: %s" % datetime.datetime.now().isoformat(timespec="seconds"))
    L.append("Benchmark: %d random %d-regular graphs on %d nodes (seed = graph index)"
             % (N_GRAPHS, DEGREE, N_NODES))
    L.append("MERGE(tau=%s) repeated %d times per graph (np seed = repeat index);" % (tag(tau), REPEATS))
    L.append("NO-MERGE is deterministic (no quantum step), shown for reference.")
    L.append("")
    L.append("%5s %8s %9s %9s %8s %8s %8s %9s %9s" %
             ("graph", "optimal", "NOMERGE", "MG_mean", "MG_std", "MG_min", "MG_max", "MG_mean_r", "exact/N"))
    for i in range(N_GRAPHS):
        s = stats[i]
        L.append("%5d %8.1f %9.4f %9.4f %8.4f %8.4f %8.4f %9.4f %7d/%d" %
                 (i, opt[i], nm[i]["cut"], s["mean"], s["std"], s["min"], s["max"],
                  s["mean_r"], s["exact_count"], REPEATS))
    L.append("")
    mr = np.mean([s["mean_r"] for s in stats])
    br = np.mean([s["best_r"] for s in stats])
    L.append("mean over graphs:  MERGE mean-ratio = %.5f   best-of-%d ratio = %.5f" % (mr, REPEATS, br))
    L.append("=" * 92)
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("Wrote", path)


def write_combined(opt, nm, per_tau, path):
    opt = np.array(opt, float)
    nmc = np.array([r["cut"] for r in nm], float)
    nm_r = nmc / opt
    nm_e = np.abs(nmc - opt)

    L = []
    L.append("=" * 100)
    L.append("DEGREE-5 DENOISED COMPARISON BY MERGE THRESHOLD  (%d repeats per graph)" % REPEATS)
    L.append("=" * 100)
    L.append("Generated: %s" % datetime.datetime.now().isoformat(timespec="seconds"))
    L.append("Benchmark: %d random %d-regular graphs on %d nodes (seed = graph index)"
             % (N_GRAPHS, DEGREE, N_NODES))
    L.append("")
    L.append("Each MERGE(tau) per-graph result is averaged over %d independent repeats" % REPEATS)
    L.append("(np seed = repeat index) to average out the shot-based QAOA merge noise.")
    L.append("Comparisons below use each graph's MEAN-over-repeats cut vs NO-MERGE.")
    L.append("")
    L.append("-" * 100)
    L.append("BASELINE  NO-MERGE  (deterministic, tau-independent)")
    L.append("-" * 100)
    L.append("  mean approx ratio        : %.6f" % nm_r.mean())
    L.append("  exact-optimal hits       : %d / %d" % (int((nm_e <= 1e-6).sum()), N_GRAPHS))
    L.append("")
    L.append("-" * 100)
    L.append("MERGE(tau)  DENOISED SUMMARY")
    L.append("-" * 100)
    L.append("%6s %10s %10s %10s %11s %11s %10s %12s %12s" %
             ("tau", "mean_r", "best_r", "worst_r", "exact_rate", "exact_any", "mean_std",
              "MERGE_wins", "NOMERGE_wins"))
    L.append("%6s %10.5f %10s %10s %11s %11s %10s %12s %12s" %
             ("none", nm_r.mean(), "-", "-", "%d/%d" % (int((nm_e <= 1e-6).sum()), N_GRAPHS),
              "-", "-", "-", "-"))
    for s in per_tau:
        L.append("%6s %10.5f %10.5f %10.5f %11s %11s %10.4f %12d %12d" %
                 (tag(s["tau"]), s["mean_r"], s["best_r"], s["worst_r"],
                  "%d/%d" % (s["exact_total"], N_GRAPHS * REPEATS),
                  "%d/%d" % (s["exact_any_graphs"], N_GRAPHS),
                  s["mean_std"], s["merge_better"], s["nomerge_better"]))
    L.append("")
    L.append("mean_r   = mean over graphs of (mean-over-repeats cut / optimal)")
    L.append("best_r   = mean over graphs of (best-over-repeats cut / optimal)  [optimistic]")
    L.append("worst_r  = mean over graphs of (worst-over-repeats cut / optimal) [pessimistic]")
    L.append("exact_rate    = fraction of (graph x repeat) runs that hit the exact optimum")
    L.append("exact_any     = #graphs where at least one repeat hit the optimum")
    L.append("mean_std      = mean over graphs of the cut std across repeats (noise level)")
    L.append("MERGE/NOMERGE_wins = head-to-head on mean cut vs NO-MERGE (|error| basis)")
    L.append("")
    best = max(per_tau, key=lambda s: (s["mean_r"]))
    L.append("Best threshold by denoised mean ratio: tau = %s (mean_r=%.5f)."
             % (tag(best["tau"]), best["mean_r"]))
    L.append("")
    L.append("-" * 100)
    L.append("PER-GRAPH MEAN-OVER-REPEATS RATIO  (NO-MERGE | MERGE mean at each tau)")
    L.append("-" * 100)
    hdr = "%5s %8s %10s" % ("graph", "optimal", "NOMERGE")
    for s in per_tau:
        hdr += " %11s" % ("MG@" + tag(s["tau"]))
    L.append(hdr)
    for i in range(N_GRAPHS):
        row = "%5d %8.1f %10.4f" % (i, opt[i], nm_r[i])
        for s in per_tau:
            row += " %11.4f" % s["pg"][i]["mean_r"]
        L.append(row)
    L.append("")
    L.append("=" * 100)
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("Wrote", path)


def main():
    graphs = build_graphs()
    opt = []
    nm = []
    for i, G in enumerate(graphs):
        opt.append(ab.quiet(cd.solver, G))
        np.random.seed(0)
        nm.append(ab.run_decompo(G.copy(), mt.M, use_merge=False))
        print("baseline graph %2d opt=%.1f NO-MERGE=%.4f" % (i, opt[i], nm[i]["cut"]))

    per_tau = []
    for tau in TAUS:
        mt.tau = tau
        stats = []
        for i, G in enumerate(graphs):
            cuts = []
            for r in range(REPEATS):
                np.random.seed(r)
                res = ab.run_decompo(G.copy(), mt.M, use_merge=True)
                cuts.append(res["cut"])
            s = per_graph_stats(cuts, opt[i])
            stats.append(s)
            print("tau=%s graph %2d MERGE mean=%.4f std=%.4f (min=%.3f max=%.3f) exact=%d/%d"
                  % (tag(tau), i, s["mean"], s["std"], s["min"], s["max"], s["exact_count"], REPEATS))
        write_tau_avg(tau, opt, nm, stats, os.path.join(HERE, "COMP_T%s_avg.txt" % tag(tau)))

        mean_r = float(np.mean([s["mean_r"] for s in stats]))
        best_r = float(np.mean([s["best_r"] for s in stats]))
        worst_r = float(np.mean([s["worst_r"] for s in stats]))
        exact_total = sum(s["exact_count"] for s in stats)
        exact_any_graphs = sum(1 for s in stats if s["exact_any"])
        mean_std = float(np.mean([s["std"] for s in stats]))
        mb = sum(1 for i, s in enumerate(stats)
                 if abs(s["mean"] - opt[i]) < abs(nm[i]["cut"] - opt[i]) - 1e-6)
        nb = sum(1 for i, s in enumerate(stats)
                 if abs(nm[i]["cut"] - opt[i]) < abs(s["mean"] - opt[i]) - 1e-6)
        per_tau.append({
            "tau": tau, "mean_r": mean_r, "best_r": best_r, "worst_r": worst_r,
            "exact_total": exact_total, "exact_any_graphs": exact_any_graphs,
            "mean_std": mean_std, "merge_better": mb, "nomerge_better": nb, "pg": stats,
        })

    write_combined(opt, nm, per_tau, os.path.join(HERE, "COMP_BY_T_REPEAT.txt"))


if __name__ == "__main__":
    main()
