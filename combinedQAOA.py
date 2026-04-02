import copy
import sympy as sp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random as rd
import math
import gurobipy as gp
from gurobipy import GRB
from qiskit.quantum_info import Pauli, Operator
from qiskit.quantum_info import SparsePauliOp as spo
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer.primitives import SamplerV2 as AerSampler
from scipy.optimize import minimize
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
import qiskit_aer as Aer


# generate sparse or dense graph
# kept from both files with only minimal cleanup

def genGraph(nodes, edges, sparceOrDense):
   if(edges == -1):
      if(sparceOrDense == "dense"):
         edges = rd.randint(math.ceil((nodes * nodes) / 2) - nodes, math.ceil(((nodes * nodes) / 2) - (nodes / 2)))
         if(edges > math.ceil(nodes * nodes / 2) - math.ceil(nodes / 2)):
            edges = math.ceil(nodes * nodes / 2) - math.ceil(nodes / 2)
      else:
         edges = rd.randint(nodes - 1, 2 * nodes)
         if(edges > math.ceil(nodes * nodes / 2) - math.ceil(nodes / 2)):
            edges = nodes - 1

   print("Edges: ", edges)
   graph = nx.Graph()
   graph.add_nodes_from(list(range(0, nodes)))
   for i in range(0, nodes - 1):
      graph.add_edge(i, i + 1)
   j = len(graph.edges())
   while j < edges:
      xOne = rd.randint(0, nodes - 1)
      xTwo = rd.randint(0, nodes - 1)
      if((not graph.has_edge(xOne, xTwo)) and xOne != xTwo):
         graph.add_edge(xOne, xTwo)
         j += 1

   return graph


# calculate single expectation vals

def zExpect(counts, i):
   totalZ = 0.0
   shots = sum(counts.values())
   for bitstring, count in counts.items():
       z_i = 1 if bitstring[-1 - i] == '0' else -1
       totalZ += count * z_i
   return totalZ / shots


# calculate double expectation vals

def zzExpect(counts, i, j):
   totalZZ = 0.0
   shots = sum(counts.values())
   for bitstring, count in counts.items():
       z_i = 1 if bitstring[-1 - i] == '0' else -1
       z_j = 1 if bitstring[-1 - j] == '0' else -1
       totalZZ += count * z_i * z_j
   return totalZZ / shots


# build Rz and Rzz for QAOA

def buildPaulis(problem, graph):
   pauliList = []
   offset = 0.0
   if(problem.lower() == "maxcut" or problem.lower() == "max cut"):
      for u, v in graph.edges():
         pauliList.append(("ZZ", [u, v], -0.5))
         offset += 0.5
      #for x in list(graph.nodes()):
      #   pauliList.append(("Z", [x], graph.degree[x]))
      return pauliList, offset

   elif(problem.lower() == "mis"):
      for x in list(graph.nodes()):
         pauliList.append(("Z", [x], graph.degree[x]))
      return pauliList

   else:
      print("new Problem?")
      return -99


# Generate polynomial for system of Eq's

def genPolyCut(graph, theC, kind):
   polyDict = {}
   if(kind):
      for (i, j) in graph.edges():
         if(("z" + str(i)) not in polyDict):
            polyDict["z" + str(i)] = 1
         else:
            polyDict["z" + str(i)] += 1

         if(("z" + str(j)) not in polyDict):
            polyDict["z" + str(j)] = 1
         else:
            polyDict["z" + str(j)] += 1

         polyDict["z" + str(i) + "z" + str(j)] = -2
   else:
      for (i, j) in graph.edges():
         if((i not in theC) and (j not in theC)):
            if(("z" + str(i)) not in polyDict):
               polyDict["z" + str(i)] = 1
            else:
               polyDict["z" + str(i)] += 1
            if(("z" + str(j)) not in polyDict):
               polyDict["z" + str(j)] = 1
            else:
               polyDict["z" + str(j)] += 1
            polyDict["z" + str(i) + "z" + str(j)] = -2
         elif((i in theC) and (j not in theC)):
            if(("z" + str(j)) not in polyDict):
               polyDict["z" + str(j)] = 1
            else:
               polyDict["z" + str(j)] += 1
            polyDict["z" + str(i) + "z" + str(j)] = -2
         elif((i not in theC) and (j in theC)):
            if(("z" + str(i)) not in polyDict):
               polyDict["z" + str(i)] = 1
            else:
               polyDict["z" + str(i)] += 1
            polyDict["z" + str(i) + "z" + str(j)] = -2

   return polyDict


# optimize for fixed F

def maxWFixed(bStr, poly, cut):
   m = gp.Model("f")
   m.setParam('OutputFlag', 0)
   indv = []
   for entry in poly:
      if(entry.count('z') == 1):
         indv.append(entry)
      elif(entry.count('z') == 2):
         sec = entry[1:].find('z')
         if(entry[:sec + 1] not in indv):
            indv.append(entry[:sec + 1])
         if(entry[sec + 1:] not in indv):
            indv.append(entry[sec + 1:])
   print(sorted(indv))
   indv = sorted(indv)
   m.addVars(len(indv), vtype=GRB.BINARY, name=indv)
   m.update()
   i = 0
   mvars = {}
   objf = 0
   for var in m.getVars():
      mvars[var.varName] = var
      if(int(var.varName[1:]) in cut):
         m.addConstr(var == int(bStr[i]))
         i += 1
         print(f"{var.varName} set as {bStr[i - 1]}")
   for entry in poly:
      if(entry.count('z') == 2):
         sec = entry[1:].find('z')
         objf += mvars[entry[:sec + 1]] * mvars[entry[sec + 1:]] * poly[entry]
      elif(entry.count('z') == 1):
         objf += mvars[entry] * poly[entry]
   print(objf)
   m.setObjective(objf, GRB.MAXIMIZE)
   m.optimize()
   return m.ObjVal


# reweight edges

def reWeight(cut, v2K, J):
   run_basic_qaoa(v2K, reps = 1, problem="maxcut", shots=40000, draw_graph=True)
   sols = []
   c = 0
   polyN = genPolyCut(v2K, cut, kind=False)
   print(polyN)
   for i in range(0, (2 ** (len(cut)))):
      binAp = str(bin(i))[2:]
      n = binAp.zfill(len(cut))
      sols.append([n, maxWFixed(n, polyN, cut)])
   print(f"SOLS : {sols}")
   return sols


# find minVertCutSet, reweight
# same logic as original decompQAOA, but uses the passed graph instead of global thisG

def decomp(tGraph, lim):
   cutSet = sorted(list(nx.minimum_node_cut(tGraph)))
   copyG = tGraph.copy()
   copyG.remove_nodes_from(cutSet)
   seps = sorted(list(nx.connected_components(copyG)), key=len, reverse=True)
   sepsv1 = []
   sepsv2 = []
   i = 0
   print(seps)
   if(len(seps) >= 2):
      for m in range(math.floor(len(seps) / 2)):
         for item in seps[m]:
            sepsv1.append(item)
      i = math.floor(len(seps) / 2)
      while(i < len(seps)):
         for item in seps[i]:
            sepsv2.append(item)
         i += 1
   print(sepsv1)
   print(sepsv2)
   print(f"cutset: {cutSet}")
   v1 = tGraph.copy()
   v1.remove_nodes_from(list(sepsv1) + cutSet)
   v2 = tGraph.copy()
   v2.remove_nodes_from(list(sepsv2) + cutSet)

   v2K = tGraph.copy()
   v2K.remove_nodes_from(list(sepsv1))

   v1K = tGraph.copy()
   v1K.remove_nodes_from(list(sepsv2))

   reWeight(cutSet, v1K, 2)

   return {
      "cutSet": cutSet,
      "seps": seps,
      "sepsv1": sepsv1,
      "sepsv2": sepsv2,
      "v1": v1,
      "v2": v2,
      "v1K": v1K,
      "v2K": v2K
   }


# QAOA helper pieces from basicQAOA.py

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
      print(f"Cost = {val:.4f}  |  params = {params}")
      return val

   result = minimize(
      wrapped_cost,
      init_params,
      method="COBYLA",
      options={"maxiter": maxiter, "disp": True}
   )

   return result, transpiled_ansatz, objective_func_vals




def run_basic_qaoa(graph, reps, problem="maxcut", shots=40000, filter_z2 = True, draw_graph=True):

   n = graph.number_of_nodes()
   code = {old: i for i, old in enumerate(graph.nodes())}
   graph = nx.relabel_nodes(graph, code)
   
   if draw_graph:
      nx.draw(graph, with_labels=True)
      plt.show()

   thePauli, shift = buildPaulis(problem, graph)
   print("Paulis: ", thePauli)

   costH = spo.from_sparse_list(thePauli, num_qubits=n)
   print("Cost Diagonals:")
   for entry in spo.to_matrix(costH):
      i = 0
      for sub in entry:
         if(str(sub)[0] == '0'):
            pass
         else:
            print(sub)
            i += 1
      if i == 0:
         print("(0.0j)")
   print("SHIFT: ", shift)

   result, transpiled_ansatz, objective_func_vals = optimize_qaoa(costH, reps=reps)
   print(f"Optimization finished. Final cost: {-1 * (result.fun) + shift :.4f}")

   plt.figure(figsize=(8, 5))
   plt.plot(objective_func_vals, "o-")
   plt.xlabel("Iteration")
   plt.ylabel("Cost (<H>)")
   plt.title("QAOA Convergence")
   plt.grid(True)
   plt.show()

   sampler = AerSampler()
   sampler.options.default_shots = shots

   optimal_circuit = transpiled_ansatz.assign_parameters(result.x)
   print(optimal_circuit)
   optimal_circuit.measure_all()
   backendFinal = Aer.AerSimulator()
   countsF = backendFinal.run(optimal_circuit, shots=shots).result().get_counts()
   countsFinal = {}

   if(filter_z2):
      for entry in countsF:
         if(sum(int(bit) for bit in entry) <= (n / 2)):
            countsFinal[entry] = countsF[entry]
   else:
      countsFinal = countsF

   print(countsFinal)

   data = countsFinal
   sorted_keys = sorted(data.keys())
   counts = [data[k] for k in sorted_keys]

   plt.figure(figsize=(11, 6))
   bars = plt.bar(sorted_keys, counts, color='orange', edgecolor='black', width=0.65)

   for bar in bars:
      height = bar.get_height()
      plt.text(
         bar.get_x() + bar.get_width() / 2,
         height + 3,
         f'{int(height)}',
         ha='center', va='bottom', fontsize=20, fontweight='bold'
      )

   plt.xlabel('Final State Measured', fontsize=25, fontweight='bold')
   plt.ylabel('Frequency of Measurement', fontsize=25, fontweight='bold')
   plt.title('Solution State Distribution', fontsize=28, pad=15, fontweight='bold')
   plt.xticks(sorted_keys, fontsize=12, fontweight='bold', rotation=45)
   plt.yticks(fontsize=19)
   plt.grid(axis='y', linestyle='--', alpha=0.4)
   if len(counts) > 0:
      plt.ylim(0, max(counts) * 1.18)
   plt.tight_layout()
   plt.show()

   if draw_graph:
      nx.draw(graph, with_labels=True)
      plt.show()

   for i in range(0, n):
      zi = zExpect(countsFinal, i)
      print(f"Node {i}: <Z{i}> = {zi:.4f}")

   for edge in graph.edges():
      i, j = edge
      zizj = zzExpect(countsFinal, i, j)
      print(f"Edge ({i},{j}): ⟨Z{i}Z{j}⟩ ≈ {zizj:.4f}")
      cut = (1 - zizj) / 2
      print(f"  → Prob this edge is cut = {cut:.4f}\n")

   return {
      "paulis": thePauli,
      "shift": shift,
      "costH": costH,
      "result": result,
      "objective_func_vals": objective_func_vals,
      "counts_raw": countsF,
      "counts_filtered": countsFinal,
   }





if __name__ == "__main__":
   # example 1: basicQAOA graph
   n = 6
   thisG = nx.Graph()
   thisG.add_nodes_from(list(range(0, n)))
   #thisG.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)])
   # example 2: decompQAOA graph
   # n = 6
   # thisG = nx.Graph()
   thisG.add_edges_from([(0,1),(0,2),(0,3),(1,4),(1,5),(2,4),(2,5),(3,4),(3,5)])
   # decomp_results = decomp(thisG, 2)
   # print(decomp_results)
   decomp(thisG, 2)

