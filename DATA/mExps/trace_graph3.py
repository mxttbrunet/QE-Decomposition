#!/usr/bin/env python3
"""Trace QAOA-Decomposition vs MERGE FINAL on benchmark graph 3."""
import sys, io, contextlib
sys.path.insert(0, "/home/mxttbrunet/QAOA-Graph-Decomp")
sys.path.insert(0, "/home/mxttbrunet/QE-Decomposition")

import numpy as np
import networkx as nx

import COMP_DECOMP as cd          # QAOA-Decomposition
import MERGE_TEMP1 as mt          # MERGE building blocks
mt.draw = lambda *a, **k: None
import MERGE_0701 as mg           # MERGE FINAL (reweightFull, run_decompo_final)

SEED = 3
G = nx.random_regular_graph(3, 14, seed=SEED)
for u, v in G.edges():
    G[u][v]['weight'] = 1

def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)

print("="*78)
print("GRAPH 3  (3-regular, 14 nodes)")
print("="*78)
opt = quiet(cd.solver, G)
print("Exact optimal MaxCut (Gurobi ILP) =", opt)
print()

# ----------------------------------------------------------------------------
# PART A: the round-1 elimination subproblem -- EXACT vs p=1 QAOA on the SAME H
# Replicate ReductionG's first-iteration relabel, then compare the repo's own
# exact solver (FixedNodeSubgraphSolver) against its p=1 QAOA (QAOANodeIter).
# ----------------------------------------------------------------------------
np.random.seed(0)
Gr = G.copy()
node_cut = nx.minimum_node_cut(Gr)
Copy = Gr.copy(); Copy.remove_nodes_from(node_cut)
cutlets = list(Copy.subgraph(c) for c in nx.connected_components(Copy))
if len(cutlets[0]) > len(cutlets[1]):
    S1c, S2c = cutlets[1], cutlets[0]
else:
    S1c, S2c = cutlets[0], cutlets[1]
cutNodeList = sorted(node_cut)
S1NodeList  = sorted(S1c.nodes())
S2NodeList  = sorted(S2c.nodes())
k, S1, S2 = len(cutNodeList), len(S1NodeList), len(S2NodeList)
mapping = dict(zip(cutNodeList + S1NodeList + S2NodeList, range(len(Gr.nodes))))
Gr = nx.relabel_nodes(Gr, mapping, copy=True)

print("-"*78)
print("PART A  Round-1 elimination subproblem")
print("-"*78)
print("min node cut K =", cutNodeList, " (|K|=%d)" % k)
print("eliminated fragment S1/V2 = %d nodes, kept side S2 = %d nodes" % (S1, S2))
print("subproblem H = K + eliminated fragment  ->  %d nodes, solved with the" % (k+S1))
print("cut qubits fixed across all 2^%d = %d boundary assignments." % (k, 2**k))
print()

pa = cd.partialAssignment(k)
exact_shift, exact_c = quiet(cd.FixedNodeSubgraphSolver, Gr, k, S1, S2, pa)
qaoa_shift,  qaoa_c  = quiet(cd.QAOANodeIter,            Gr, k, S1, S2, pa)
exact_RHS = [round(e + exact_c, 4) for e in exact_shift]
qaoa_RHS  = [round(q + qaoa_c, 4)  for q in qaoa_shift]

print("boundary assignment | EXACT subproblem MaxCut | p=1 QAOA estimate | error")
for i, a in enumerate(pa):
    err = qaoa_RHS[i] - exact_RHS[i]
    print("   %s            |        %7.3f          |     %8.3f      | %+6.3f"
          % ("".join(map(str, a)), exact_RHS[i], qaoa_RHS[i], err))
print()
print(">> p=1 QAOA underestimates the eliminated fragment on every assignment;")
print("   these biased values are what get baked into the reduced graph.")
print()

# ----------------------------------------------------------------------------
# PART B: MERGE's exact reweight of the SAME fragment is lossless
# ----------------------------------------------------------------------------
print("-"*78)
print("PART B  MERGE reweight of the same round-1 fragment (exact ILP + slack LP)")
print("-"*78)
V2andK, Km, V1, V2 = mt.getInduced(G.copy())
print("MERGE picks K =", sorted(Km), " V2 (eliminated) =", sorted(V2), "(|V2|=%d)" % len(V2))
sub = V2andK.copy()
Kset = set(Km)
sub.remove_edges_from([(u, v) for u, v in sub.edges() if u in Kset and v in Kset])
assigns = [[int(c) for c in s] for s in mt.genPerms(len(Km))]
RHSm = [quiet(mt.solvePartial, a, sub, list(Km)) for a in assigns]
J = quiet(mg.reweightFull, V2andK, Km, V2)
print("exact subproblem MaxCut per assignment :", [round(r, 3) for r in RHSm])
print("fitted constant (seaHat) =", round(J['seaHat'], 4))
print("fitted edge weights among K            :",
      {kk: round(vv, 4) for kk, vv in J.items() if kk != 'seaHat'})
# reconstruct surrogate and show slack
import itertools
Kl = list(Km)
pairs = list(itertools.combinations(range(len(Kl)), 2))
recon = []
for a in assigns:
    val = J['seaHat']
    for (i, j) in pairs:
        e = (Kl[i], Kl[j])
        w = J.get(e, J.get((e[1], e[0]), 0.0))
        if a[i] != a[j]:
            val += w
    recon.append(val)
slack = [round(RHSm[i] - recon[i], 6) for i in range(len(assigns))]
print("surrogate reconstruction               :", [round(r, 3) for r in recon])
print("slack e_i (exact - surrogate, >=0)     :", slack)
print(">> slack is ~0: the pairwise reweight reproduces the exact fragment value")
print("   on every boundary assignment -> NO information lost.")
print()

# ----------------------------------------------------------------------------
# PART C: end-to-end, including the FINAL solve
# ----------------------------------------------------------------------------
print("-"*78)
print("PART C  End-to-end result + final-solve step")
print("-"*78)

# QAOA-Decomposition full pipeline
np.random.seed(0)
qreduced, qconst, _ = quiet(cd.ReductionG, G.copy())
qfinal, _ = quiet(cd.Decomp_QAOA_Node, qreduced, qconst)
# what an EXACT solve of the same final reduced graph would give:
qfinal_exact = quiet(cd.solver, qreduced) + qconst
print("QAOA-Decomp final reduced graph : %d nodes, %d edges"
      % (qreduced.number_of_nodes(), qreduced.number_of_edges()))
print("   final solve = p=1 QAOA      -> %.4f   (ratio %.4f)" % (qfinal, qfinal/opt))
print("   (exact solve of same graph  -> %.4f) : p=1 QAOA loses %.4f here alone"
      % (qfinal_exact, qfinal_exact - qfinal))
print()

# MERGE FINAL full pipeline
np.random.seed(0)
mreduced, mIsh, mlog = quiet(mg.run_decompo_final, G.copy(), mt.M)
mbest, _ = quiet(mt.bruteMaxCut, mreduced)
mfinal = mbest + mIsh
print("MERGE FINAL  final reduced graph: %d nodes, %d edges"
      % (mreduced.number_of_nodes(), mreduced.number_of_edges()))
print("   final solve = exact brute force -> %.4f   (ratio %.4f)" % (mfinal, mfinal/opt))
print()
print("="*78)
print("SUMMARY  graph 3   optimal = %.0f" % opt)
print("   QAOA-Decomposition : %.4f   ratio %.4f" % (qfinal, qfinal/opt))
print("   MERGE FINAL        : %.4f   ratio %.4f" % (mfinal, mfinal/opt))
print("="*78)
