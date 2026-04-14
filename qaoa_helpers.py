"""
qaoa_helpers.py

QAOA infrastructure and graph utility functions shared across QIRA modules.
Originally from recDiv.py; kept identical in functionality.
"""

import math
import random as rd

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import qiskit_aer as Aer
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp as spo
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit.transpiler import generate_preset_pass_manager
from scipy.optimize import minimize


def buildPaulis(problem, graph):
    #Build Pauli list for the QAOA cost Hamiltonian.
    pauliList = []
    offset = 0.0
    if problem.lower() in ("maxcut", "max cut"):
        for u, v in graph.edges():
            w = graph[u][v].get('weight', 1.0)
            pauliList.append(("ZZ", [u, v], -w / 2))
            offset += 0.5 * w
        # Linear (diagonal) QUBO terms produced by the decomp reweighting step.
        # In the Ising mapping z_i = (1 - Z_i)/2:
        #   J_ii * z_i  →  -J_ii/2 * Z_i  +  J_ii/2  (constant)
        for v in graph.nodes():
            nw = graph.nodes[v].get('weight', 0.0)
            if abs(nw) > 1e-12:
                pauliList.append(("Z", [v], -nw / 2))
                offset += nw / 2
        return pauliList, offset
    else:
        print("new Problem?")
        return -99


def zExpect(counts, i):
    totalZ = 0.0
    shots = sum(counts.values())
    for bitstring, count in counts.items():
        z_i = 1 if bitstring[-1 - i] == '0' else -1
        totalZ += count * z_i
    return totalZ / shots


def zzExpect(counts, i, j):
    totalZZ = 0.0
    shots = sum(counts.values())
    for bitstring, count in counts.items():
        z_i = 1 if bitstring[-1 - i] == '0' else -1
        z_j = 1 if bitstring[-1 - j] == '0' else -1
        totalZZ += count * z_i * z_j
    return totalZZ / shots


def cost_func(params, transpiled_ansatz, hamiltonian, estimator):
    bound_circuit = transpiled_ansatz.assign_parameters(params)
    pub = (bound_circuit, hamiltonian)
    job = estimator.run([pub])
    result = job.result()[0]
    return -result.data.evs


def optimize_qaoa(costH, reps, maxiter=150):
    qaoa_ansatz = QAOAAnsatz(cost_operator=costH, reps=reps)
    pm = generate_preset_pass_manager(
        optimization_level=1,
        backend=None,
        basis_gates=["u", "cx", "rz", "sx"],
    )
    transpiled_ansatz = pm.run(qaoa_ansatz)
    init_params = np.array([np.pi / 2] * reps + [np.pi] * reps)
    objective_func_vals = []
    estimator = AerEstimator()

    def wrapped_cost(params):
        val = cost_func(params, transpiled_ansatz, costH, estimator)
        objective_func_vals.append(val)
        #print(f"Cost = {val:.4f}  |  params = {params}")
        return val

    result = minimize(
        wrapped_cost,
        init_params,
        method="COBYLA",
        options={"maxiter": maxiter, "disp": False}
    )
    return result, transpiled_ansatz, objective_func_vals


def run_basic_qaoa(graph, reps, problem="maxcut", shots=40000,
                   filter_z2=False, draw_graph=False):
    n = graph.number_of_nodes()
    # Relabel to contiguous 0..n-1 for the QAOA circuit; preserve attributes.
    code  = {old: i for i, old in enumerate(graph.nodes())}
    code2 = {i: old for i, old in enumerate(graph.nodes())}
    graph_r = nx.relabel_nodes(graph, code)

    if draw_graph:
        nx.draw(graph_r, with_labels=True)
        plt.show()

    thePauli, shift = buildPaulis(problem, graph_r)
    #print("Paulis:", thePauli)

    costH = spo.from_sparse_list(thePauli, num_qubits=n)
    #print("SHIFT:", shift)

    result, transpiled_ansatz, objective_func_vals = optimize_qaoa(costH, reps=reps)
    #print(f"Optimization finished. Final cost: {-1 * result.fun + shift:.4f}")

    sampler = AerSampler()
    sampler.options.default_shots = shots

    optimal_circuit = transpiled_ansatz.assign_parameters(result.x)
    optimal_circuit.measure_all()
    backendFinal = Aer.AerSimulator()
    countsF = backendFinal.run(optimal_circuit, shots=shots).result().get_counts()
    countsFinal = {}

    if filter_z2:
        for entry in countsF:
            if sum(int(bit) for bit in entry) <= (n / 2):
                countsFinal[entry] = countsF[entry]
    else:
        countsFinal = countsF

    print(countsFinal)

    singExp = {}
    doubExp = {}
    for i in graph_r.nodes():
        singExp[code2[i]] = zExpect(countsFinal, i)
    for u, v in graph_r.edges():
        doubExp[(code2[u], code2[v])] = zzExpect(countsFinal, u, v)

    return singExp, doubExp


def genGraph(nodes, edges, sparceOrDense):
    if edges == -1:
        if sparceOrDense == "dense":
            edges = rd.randint(math.ceil((nodes * nodes) / 2) - nodes,
                               math.ceil(((nodes * nodes) / 2) - (nodes / 2)))
            if edges > math.ceil(nodes * nodes / 2) - math.ceil(nodes / 2):
                edges = math.ceil(nodes * nodes / 2) - math.ceil(nodes / 2)
        else:
            edges = rd.randint(nodes - 1, 2 * nodes)
            if edges > math.ceil(nodes * nodes / 2) - math.ceil(nodes / 2):
                edges = nodes - 1

    print("Edges:", edges)
    graph = nx.Graph()
    graph.add_nodes_from(list(range(nodes)))
    for i in range(nodes - 1):
        graph.add_edge(i, i + 1)
    j = len(graph.edges())
    while j < edges:
        a = rd.randint(0, nodes - 1)
        b = rd.randint(0, nodes - 1)
        if not graph.has_edge(a, b) and a != b:
            graph.add_edge(a, b)
            j += 1
    return graph


def makeCustom(stringy, numN):
    newg = nx.Graph()
    newg.add_nodes_from(list(range(numN)))
    i = 0
    tempE = []
    edgesN = []
    for char in stringy:
        if char.isdigit():
            i += 1
            tempE.append(int(char) - 1)
        if i == 2:
            edgesN.append((tempE[0], tempE[1], 1))
            tempE.clear()
            i = 0
    newg.add_weighted_edges_from(edgesN)
    return newg
