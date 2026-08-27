
import math
import time
import datetime
from itertools import chain

import numpy as np
import networkx as nx
from scipy.optimize import minimize
from gurobipy import *

np.random.seed(0)


def perms(n):
    if not n:
        return
    for i in range(2 ** n):
        s = bin(i)[2:]
        s = "0" * (n - len(s)) + s
        yield s


def stringArrayconvertor(arr):
    a = []
    for string in arr:
        a.append([int(j) for j in string])
    return a


def partialAssignment(n):
    return stringArrayconvertor(perms(n))


def partialAssignMixer(n, d):
    strings = list(perms(n))
    dummystrings = list(perms(d))
    if len(dummystrings) == 0:
        return partialAssignment(n)
    a = []
    for i in strings:
        for k in range(2 ** d):
            string = i + dummystrings[k]
            a.append([int(j) for j in string])
    return a


def NodeWeightGen(n_nodes):
    return np.zeros(n_nodes)


def nxnEdgeWeightArray(G, n_nodes):
    nxnEdgeArray = np.zeros((n_nodes, n_nodes))
    for (i, j) in G.edges:
        val = float(G[i][j]['weight'])
        nxnEdgeArray[i][j] = val
        nxnEdgeArray[j][i] = nxnEdgeArray[i][j]
    return nxnEdgeArray


class Function():
    def __init__(self, h, J, constant, G):
        self.G = G
        self.h = h
        self.J = J
        self.const = constant

    def func(self, angles):
        G = self.G
        h = self.h
        J = self.J

        beta = angles[0]
        gamma = angles[1]

        loc = []
        for i in G.nodes:
            c_i = h[i] * np.sin(2 * beta) * np.sin(2 * gamma * h[i])
            for k in G[i]:
                c_i *= np.cos(2 * gamma * J[i][k])
            loc.append(c_i)

        # C_ij expectation values
        coup = []
        for (i, j) in G.edges:
            c_ij_1 = (J[i][j] * np.sin(4 * beta) * np.sin(2 * gamma * J[i][j])) / 2

            c_ij_2 = np.cos(2 * gamma * h[i])
            for k in G[i]:
                if k != j:
                    c_ij_2 *= np.cos(2 * gamma * J[i][k])

            c_ij_3 = np.cos(2 * gamma * h[j])
            for k in G[j]:
                if k != i:
                    c_ij_3 *= np.cos(2 * gamma * J[j][k])

            c_ij_4 = (J[i][j] * (np.sin(2 * beta)) ** 2) / 2
            for k in G[i]:
                if k not in G[j] and k != j:
                    c_ij_4 *= np.cos(2 * gamma * J[i][k])
            for k in G[j]:
                if k not in G[i] and k != i:
                    c_ij_4 *= np.cos(2 * gamma * J[j][k])

            c_ij_5 = np.cos(2 * gamma * (h[i] + h[j]))
            for k in G[i]:
                if k in G[j]:
                    c_ij_5 *= np.cos(2 * gamma * (J[i][k] + J[j][k]))

            c_ij_6 = np.cos(2 * gamma * (h[i] - h[j]))
            for k in G[i]:
                if k in G[j]:
                    c_ij_6 *= np.cos(2 * gamma * (J[i][k] - J[j][k]))

            c_ij = (c_ij_1 * (c_ij_2 + c_ij_3) - c_ij_4 * (c_ij_5 - c_ij_6))
            coup.append(c_ij)

        func = sum(loc) + sum(coup)
        return func 


class GraphClass():
    def __init__(self, G, zero_qubits, one_qubits):
        self.G = G
        self.n = len(self.G.nodes)
        self.m = len(self.G.edges())
        self.nodeWeights = NodeWeightGen(self.n)
        self.edgeWeights = nxnEdgeWeightArray(self.G, self.n)

        self.zero_qubits = zero_qubits
        self.one_qubits = one_qubits

        self.fix0 = set(self.zero_qubits)
        self.fix1 = set(self.one_qubits)

        self.p = 1
        self.n_optimizations = 100
        self.one_iteration = 1
        self.C_best = -10 ** 10

        self.angles = np.pi * 2.0 * (np.random.uniform(size=2) - 0.5)
        self.zangles = np.zeros(2 * 1, dtype=np.float64)

    def fix_QUBO(self):
        G = self.G
        h = self.nodeWeights
        J = self.edgeWeights

        fix0 = self.zero_qubits
        fix1 = self.one_qubits
        n = self.n

        h_temp = h.copy()
        J_temp = J.copy()
        const = 0
        for i in fix1:
            const += h[i]

        for i in fix1:
            for j in fix0:
                const += J[i][j]
        for i in fix1:
            for j in G[i]:
                if j not in fix0 and j not in fix1:
                    const += J[i][j]
                    h_temp[j] -= J[i][j]
        for j in fix0:
            for i in G[j]:
                if i not in fix0 and i not in fix1:
                    h_temp[i] += J[i][j]

        for i in fix1.union(fix0):
            J_temp[i, :] = np.zeros(n)
            J_temp[:, i] = np.zeros(n)
            h_temp[i] = 0
        return h_temp, J_temp, const

    def optimize(self, angles=0):
        runs = self.n_optimizations
        obj = 10 ** 10
        h_temp, J_temp, constant = self.fix_QUBO()
        FunctionClass = Function(-h_temp / 2, J_temp / 2, constant, self.G)
        weights = J_temp.sum() / 2

        opt_angles = self.angles
        for i in range(runs):
            if runs > 1:
                angles = np.pi * 2.0 * (np.random.uniform(size=2) - 0.5)

            result = minimize(FunctionClass.func, angles, method='l-bfgs-b')
            if result.fun < obj:
                obj = result.fun
                opt_angles = result.x

        actualSol = weights / 2 + h_temp.sum() / 2 - obj + constant
        return actualSol, opt_angles


# Exact MaxCut via Gurobi
def solver(G):
    m = Model("mip1")
    m.setParam("OutputFlag", 0)
    x = {}
    y = {}
    for i in list(G.nodes):
        x[i] = m.addVar(vtype=GRB.BINARY, name='x' + str(i))
    for j in list(G.edges):
        y[j] = m.addVar(vtype=GRB.BINARY, obj=G[j[0]][j[1]]['weight'], name='y' + str(j))
    m.modelSense = GRB.MAXIMIZE
    m.update()
    for j in list(G.edges):
        m.addConstr(y[j] <= x[j[0]] + x[j[1]])
        m.addConstr(y[j] <= 2 - x[j[0]] + - x[j[1]])
        m.addConstr(y[j] >= x[j[0]] - x[j[1]])
        m.addConstr(y[j] >= x[j[1]] - x[j[0]])
    m.update()
    m.optimize()
    return m.getObjective().getValue()


def FixedNodeSolverLoop(G, k, partialAssigning):
    m = Model("MaxCut")
    GNodes = list(G.nodes())
    set_of_cut = GNodes[:k]
    length = int(len(partialAssigning))

    G_prime = G.copy()
    m.modelSense = GRB.MAXIMIZE
    m.update()

    b = []
    loop_size = 0
    constant = 0
    while loop_size < length:
        m.update()
        m.setParam('OutputFlag', 0)
        n = m.copy()
        x = {}
        y = {}
        for i in list(G_prime.nodes):
            x[i] = n.addVar(vtype=GRB.BINARY, name='x' + str(i))
        for j in list(G_prime.edges):
            y[j] = n.addVar(vtype=GRB.BINARY, obj=G_prime[j[0]][j[1]]['weight'], name='y' + str(j))
        n.update()
        for j in list(G_prime.edges):
            n.addConstr(y[j] <= x[j[0]] + x[j[1]])
            n.addConstr(y[j] <= 2 - x[j[0]] + - x[j[1]])
            n.addConstr(y[j] >= x[j[0]] - x[j[1]])
            n.addConstr(y[j] >= x[j[1]] - x[j[0]])
        p = 0
        for j in set_of_cut:
            n.addConstr(x[j] == partialAssigning[loop_size][p])
            p += 1
        n.update()
        n.optimize()
        obj_val = n.getObjective().getValue()
        n.update()
        b.append(obj_val)
        if loop_size == 0:
            constant = obj_val
        loop_size += 1
    b[:] = [ele - constant for ele in b]
    return b, constant


def FixedNodeSubgraphSolver(G, k, S1, S2, partialAssigning):
    H = G.subgraph([i for i in range(k + S1)])
    return FixedNodeSolverLoop(H, k, partialAssigning)


def QAOANodeIter(G, k, S1, S2, partialAssignment_):
    H = G.subgraph([i for i in range((k + S1))])
    template = GraphClass(H, set(), set())

    K = G.subgraph([i for i in range(k)])
    K_Nodes = list(K.nodes)

    ListofOnes = []
    for i in range(len(partialAssignment_)):
        temp = partialAssignment_[i]
        tempy = [ii for ii in range(len(temp)) if temp[ii] == 1]
        ListofOnes.append(tempy)
    ListofZeros = []
    for i in range(len(ListofOnes)):
        c = [x for x in K_Nodes if x not in ListofOnes[i]]
        ListofZeros.append(c)

    b = []
    const_val = 0
    for i in range(len(partialAssignment_)):
        together = ListofZeros[i] + ListofOnes[i]
        if len(together) == k:
            template.zero_qubits = set(ListofZeros[i])
            template.one_qubits = set(ListofOnes[i])
            tempval, ignore = template.optimize()
            if i == 0:
                const_val = tempval
            b.append(tempval)
    b[:] = [ele - const_val for ele in b]
    return b, const_val


def QAOASP1(G, k, S1, S2, d_nodes=0):
    kSetPartialAssign = partialAssignment(k)
    RHS, const_val = QAOANodeIter(G, k, S1, S2, kSetPartialAssign)

    d_partialAssignments = 2 ** (d_nodes)
    Q = nx.complete_graph(k + d_nodes)
    QNodes = list(Q.edges)

    model = Model()
    model.setParam('OutputFlag', 0)
    w = {}
    e = {}
    for i in range((d_partialAssignments) * (2 ** k)):
        e[i] = model.addVar(vtype=GRB.CONTINUOUS, name='e' + str(i))
    for j in range(len(Q.edges)):
        w[j] = model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name='w' + str(j))
    c = []
    dummykSetPartialAssign = partialAssignMixer(k, d_nodes)
    PosOFOnes = []
    for i in range(len(dummykSetPartialAssign)):
        temp = dummykSetPartialAssign[i]
        tempy = [ii for ii in range(len(temp)) if temp[ii] == 1]
        PosOFOnes.append(tempy)

        for ix in range(len(PosOFOnes)):
            row = PosOFOnes[ix]
            temp2 = []
            for edge in QNodes:
                if (edge[0] in row) and (edge[1] in row):
                    temp2.append(0)
                    continue
                if (edge[0] not in row) and (edge[1] not in row):
                    temp2.append(0)
                    continue
                else:
                    temp2.append(1)
                    continue
        c.append(temp2)

    expr = sum(e[i] for i in range((d_partialAssignments) * (2 ** k)))
    model.setObjective(sense=GRB.MINIMIZE, expr=expr)
    model.update()
    model.addConstrs((e[i] + quicksum(c[i][j] * w[j] for j in range(len(w))) ==
                      RHS[math.floor(i / (d_partialAssignments))] for i in range(len(e))))
    model.update()
    model.optimize()
    obj = model.getObjective()

    x = {}
    for v in model.getVars():
        x[v.varName] = v.x

    return x, obj, const_val


def ReductionG(Graph, d_nodes=0):
    G = Graph.copy()
    constant = 0
    Iteration = 0
    cut_log = []
    while len(list(nx.minimum_node_cut(G))) > 0:
        Iteration += 1
        node_cut = nx.minimum_node_cut(G)
        cut_nodes = sorted(node_cut)
        print("\nIteration:", Iteration, " node cut:", cut_nodes)

        Copy = G.copy()
        Copy.remove_nodes_from(node_cut)
        cutlets = list(Copy.subgraph(c) for c in nx.connected_components(Copy))
        if len(cutlets) == 1:
            print("  cut set does not split the graph into two components")
            cut_log.append((Iteration, cut_nodes, "stop: cut does not split graph"))
            break
        if len(cutlets[0]) > len(cutlets[1]):
            S1_component = cutlets[1]
            S2_component = cutlets[0]
        else:
            S2_component = cutlets[1]
            S1_component = cutlets[0]
        cutset = G.subgraph(node_cut)
        cutNodeList = sorted(list(cutset.nodes()))
        S1NodeList = sorted(list(S1_component.nodes()))
        S2NodeList = sorted(list(S2_component.nodes()))
        if len(cutNodeList) > 6:
            print("  cut set is larger than 6")
            cut_log.append((Iteration, cut_nodes, "stop: cut set larger than 6"))
            break
        if len(G.nodes) <= 2:
            print("  only two nodes are here")
            cut_log.append((Iteration, cut_nodes, "stop: <= 2 nodes"))
            break

        k = len(cutNodeList)
        S1 = len(S1NodeList)
        S2 = len(S2NodeList)
        mappingG = dict(zip([i for i in cutNodeList + S1NodeList + S2NodeList],
                            range(0, len(list(G.nodes)))))
        G = nx.relabel_nodes(G, mappingG, copy=True)
        newCutSet = [i for i in range(k)]
        CopyGraph = G.copy()
        CopyGraph.remove_nodes_from(newCutSet)
        components = list(CopyGraph.subgraph(c) for c in nx.connected_components(CopyGraph))
        if len(components) == 1:
            cut_log.append((Iteration, cut_nodes, "stop: one component after cut"))
            break

        cut_log.append((Iteration, cut_nodes,
                        "reduced (|cut|=%d, |S1|=%d, |S2|=%d)" % (k, S1, S2)))
        Variables, ObjectValue, const_val = QAOASP1(G, k, S1, S2)
        constant += const_val
        edgeWeightIndexer = (2 ** d_nodes) * (2 ** k)
        weightVariables = dict(list(Variables.items())[edgeWeightIndexer:])

        C = nx.complete_graph(d_nodes + k)
        merged = list(chain(range(0, k), range(k + S1, k + S1 + S2)))
        H = G.subgraph([i for i in merged])

        CEdges = list(C.edges)
        Graph = nx.compose(C, H)

        for i in range(len(CEdges)):
            weightVariables[CEdges[i]] = weightVariables.pop('w' + str(i))
        for edge in weightVariables.keys():
            if G.has_edge(edge[0], edge[0]) == True:
                Graph[edge[0]][edge[1]]['weight'] = weightVariables[edge]
            if G.has_edge(edge[0], edge[0]) == False:
                Graph.add_edge(edge[0], edge[1], weight=weightVariables[edge])

        for edge in list(Graph.edges()):
            if Graph[edge[0]][edge[1]]['weight'] == 0:
                Graph.remove_edge(edge[0], edge[1])

        mapping = dict(zip(Graph, range(0, k + d_nodes + S2)))
        G = nx.relabel_nodes(Graph, mapping)

    return G, constant, cut_log


def Decomp_QAOA_Node(G, constant):
    classGraph = GraphClass(G, set(), set())
    sub_opt, sub_opt_angles = classGraph.optimize()
    opt_sol = sub_opt + constant
    return opt_sol, sub_opt_angles


N_GRAPHS = 30
N_NODES = 14
DEGREE = 3


def build_graphs():
    graphs = []
    for i in range(N_GRAPHS):
        G = nx.random_regular_graph(DEGREE, N_NODES, seed=i)
        for u, v in G.edges():
            G[u][v]['weight'] = 1
        graphs.append(G)
    return graphs


def main():
    graphs = build_graphs()

    results = []  # one dict per graph
    for idx, G in enumerate(graphs):
        print("\n" + "#" * 60)
        print("GRAPH %d / %d : %d nodes, %d edges"
              % (idx, N_GRAPHS, G.number_of_nodes(), G.number_of_edges()))

        optimal = solver(G)

        t0 = time.perf_counter()
        reduced_G, constant, cut_log = ReductionG(G)
        decomp_result, _ = Decomp_QAOA_Node(reduced_G, constant)
        elapsed = time.perf_counter() - t0

        approx_ratio = decomp_result / optimal if optimal != 0 else float('nan')

        results.append({
            "idx": idx,
            "edges": sorted(tuple(sorted(e)) for e in G.edges()),
            "optimal": optimal,
            "decomp": decomp_result,
            "ratio": approx_ratio,
            "time": elapsed,
            "constant": constant,
            "reduced_nodes": reduced_G.number_of_nodes(),
            "reduced_edges": reduced_G.number_of_edges(),
            "cut_log": cut_log,
        })
        print("  result=%.6f  optimal=%.1f  ratio=%.6f  time=%.2fs"
              % (decomp_result, optimal, approx_ratio, elapsed))

    write_report(results)


def write_report(results):
    ratios = [r["ratio"] for r in results]
    times = [r["time"] for r in results]

    lines = []
    lines.append("=" * 70)
    lines.append("DECOMPOSITION ALGORITHM ANALYSIS")
    lines.append("(QAOA graph-decomposition algorithm in this directory)")
    lines.append("=" * 70)
    lines.append("Generated: %s" % datetime.datetime.now().isoformat(timespec="seconds"))
    lines.append("Batch: %d random %d-regular graphs on %d nodes (seed = graph index)"
                 % (N_GRAPHS, DEGREE, N_NODES))
    lines.append("")

    lines.append("-" * 70)
    lines.append("AGGREGATE SUMMARY")
    lines.append("-" * 70)
    lines.append("Graphs run            : %d" % len(results))
    lines.append("Approximation ratio   : mean=%.6f  min=%.6f  max=%.6f  std=%.6f"
                 % (float(np.mean(ratios)), float(np.min(ratios)),
                    float(np.max(ratios)), float(np.std(ratios))))
    lines.append("Time per graph (s)    : mean=%.4f  min=%.4f  max=%.4f  total=%.4f"
                 % (float(np.mean(times)), float(np.min(times)),
                    float(np.max(times)), float(np.sum(times))))
    lines.append("")

    lines.append("-" * 70)
    lines.append("PER-GRAPH RESULTS")
    lines.append("-" * 70)
    lines.append("%5s %14s %10s %14s %12s" %
                 ("graph", "decomp_maxcut", "optimal", "approx_ratio", "time_s"))
    for r in results:
        lines.append("%5d %14.6f %10.1f %14.6f %12.4f" %
                     (r["idx"], r["decomp"], r["optimal"], r["ratio"], r["time"]))
    lines.append("")

    lines.append("=" * 70)
    lines.append("PER-GRAPH DETAIL (cut set for each decomposition iteration)")
    lines.append("=" * 70)
    for r in results:
        lines.append("")
        lines.append("GRAPH %d" % r["idx"])
        lines.append("  Edge list: %s" % ", ".join("(%d,%d)" % e for e in r["edges"]))
        lines.append("  Decomposition iterations:")
        if not r["cut_log"]:
            lines.append("    (no node cut found / no iterations)")
        else:
            for it, cut_nodes, outcome in r["cut_log"]:
                lines.append("    iter %2d : cut set = %s  ->  %s"
                             % (it, "{" + ", ".join(str(c) for c in cut_nodes) + "}", outcome))
        lines.append("  Reduced graph : %d nodes, %d edges"
                     % (r["reduced_nodes"], r["reduced_edges"]))
        lines.append("  Additive constant : %.10f" % r["constant"])
        lines.append("  Decomp MaxCut result : %.10f" % r["decomp"])
        lines.append("  Optimal MaxCut       : %.10f" % r["optimal"])
        lines.append("  Approximation ratio  : %.10f" % r["ratio"])
        lines.append("  Time taken (s)       : %.6f" % r["time"])
    lines.append("")
    lines.append("=" * 70)

    report = "\n".join(lines) + "\n"
    with open("DEC_ANALYSIS.txt", "w") as f:
        f.write(report)
    print("\nWrote DEC_ANALYSIS.txt (%d graphs)" % len(results))


if __name__ == "__main__":
    main()
