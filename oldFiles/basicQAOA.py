import sympy as sp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random as rd
import math
from qiskit.quantum_info import Pauli, Operator
from qiskit.quantum_info import SparsePauliOp as spo
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from scipy.optimize import minimize
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.primitives import Estimator
import qiskit_aer as Aer



#generate sparse or dense graph
def genGraph(nodes, edges, sparceOrDense):
   if(edges == -1):
      if(sparceOrDense == "dense"):
         edges = rd.randint( math.ceil((nodes*nodes)/2) - nodes, math.ceil( ((nodes*nodes) / 2) - (nodes / 2)) )
         if(edges > math.ceil(nodes * nodes / 2) - math.ceil(nodes / 2)):
            edges = math.ceil(nodes * nodes / 2) - math.ceil(nodes / 2)
      else:
         edges = rd.randint(nodes - 1, 2 * nodes)
         if(edges > math.ceil(nodes * nodes / 2) - math.ceil(nodes / 2)):
            edges = nodes - 1

   print("Edges: ", edges)
   graph = nx.Graph()
   graph.add_nodes_from(list(range(0,nodes)))
   for i in range(0,nodes - 1):
      graph.add_edge(i, i + 1)
   j = len(graph.edges())
   while j < (edges):
      xOne = rd.randint(0,nodes-1)
      xTwo = rd.randint(0,nodes-1)
      if( not (graph.has_edge(xOne,xTwo)) and xOne != xTwo ):
         graph.add_edge(xOne,xTwo)
         j+=1

   return graph
     


#get single expectation values
def zExpect(counts, i):
   totalZ = 0.0
   shots = sum(counts.values())
   for bitstring, count in counts.items():
       z_i = 1 if bitstring[-1-i] == '0' else -1
       #z_j = 1 if bitstring[-1-j] == '0' else -1
       totalZ += count * z_i
   return totalZ / shots



#get double expectation value 
def zzExpect(counts, i,j):
   totalZZ = 0.0
   shots = sum(counts.values())
   for bitstring, count in counts.items():
       z_i = 1 if bitstring[-1-i] == '0' else -1
       z_j = 1 if bitstring[-1-j] == '0' else -1
       totalZZ += count * z_i * z_j
   return totalZZ / shots



#Get Rzz Gates for QAOA, Hc
def buildPaulis(problem, graph):
   pauliList = []
   offset = 0.0
   if(problem.lower() == "maxcut" or problem.lower() == "max cut"):

      for u,v in graph.edges():
         pauliList.append(("ZZ", [u,v], -0.5))
         offset+=0.5
      #pauliList.sort(key=lambda x: (-max(x[1]), x[1]))
      return pauliList, offset

   elif(problem.lower() == "mis"):
      for x in list(graph.nodes()):
         pauliList.append(("Z", [x], 2)) 
      return pauliList

   else:
      print("new Problem?")
      return -99





#main code --------------------

n = 6

#thisG = genGraph(n, -1, "sparse")  #generate random dense or sparse graph
thisG = nx.Graph()

## // add graph here or use genGraph
thisG.add_nodes_from(list(range(0,n)))
thisG.add_edges_from([(0,1),(0,2),(0,3),(1,4),(1,5),(2,4),(2,5),(3,4), (3,5)])

#thisG.add_edges_from([(0,1),(0,2),(0,3)]) #,(1,2),(2,3),(3,4)])
nx.draw(thisG, with_labels = True)
plt.show()



thePauli, shift = buildPaulis("maxcut", thisG)
print("Paulis: ", thePauli)

costH = spo.from_sparse_list(thePauli, num_qubits = n)
print("Cost Diagonals:")
for entry in spo.to_matrix(costH):
   i = 0
   for sub in entry:
      if(str(sub)[0] == '0'):
         pass
      else:
         (print(sub))
         i+=1
   if i==0:
      print("(0.0j)")
print("SHIFT: ",shift )

#P LAYER ADJUSTMENT
reps = 1

#make qaoa circuit
qaoa_ansatz = QAOAAnsatz(cost_operator=costH, reps=reps)
pm = generate_preset_pass_manager(
    optimization_level=1,
    backend=None,
    basis_gates=["u", "cx", "rz", "sx"],  
)

#transpile to gate set 
transpiled_ansatz = pm.run(qaoa_ansatz)


def cost_func(params, transpiled_ansatz, hamiltonian, estimator):
    bound_circuit = transpiled_ansatz.assign_parameters(params)   # ← use transpiled_ansatz here
    pub = (bound_circuit, hamiltonian)
    job = estimator.run([pub])
    result = job.result()[0]
    return -result.data.evs

init_params = np.array([np.pi / 2] * reps + [np.pi] * reps)
objective_func_vals = []
estimator = AerEstimator()

## classically optimize params  and run QAOA
def wrapped_cost(params):
    val = cost_func(params, transpiled_ansatz, costH, estimator)
    objective_func_vals.append(val)
    print(f"Cost = {val:.4f}  |  params = {params}")
    return val

result = minimize(
    wrapped_cost,
    init_params,
    method="COBYLA",
    options={"maxiter": 120, "disp": True}
)
print(f"Optimization finished. Final cost: {-1 * (result.fun) + shift :.4f}")

# Plot convergence
plt.figure(figsize=(8, 5))
plt.plot(objective_func_vals, "o-")
plt.xlabel("Iteration")
plt.ylabel("Cost (<H>)")
plt.title("QAOA Convergence")
plt.grid(True)
plt.show()

sampler = AerSampler()
sampler.options.default_shots = 40000

optimal_circuit = transpiled_ansatz.assign_parameters(result.x)
print(optimal_circuit)
optimal_circuit.measure_all()
backendFinal = Aer.AerSimulator()
countsF = backendFinal.run(optimal_circuit,shots = 40000).result().get_counts()
countsFinal = {}

#because Z2 symmetric, look at half 
for entry in countsF:
   if(sum(int(bit) for bit in entry) <= (n / 2)):
      countsFinal[entry] = countsF[entry]

#get counts, plot 
print(countsFinal)

data = countsFinal
sorted_keys = sorted(data.keys())
counts = [data[k] for k in sorted_keys]

plt.figure(figsize=(11, 6))

bars = plt.bar(sorted_keys, counts, color='orange', edgecolor='black', width=0.65)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,    
        height + 3,                           
        f'{int(height)}',                     
        ha='center', va='bottom', fontsize=20, fontweight='bold'
    )

plt.xlabel('Final State Measured', fontsize=25, fontweight='bold')
plt.ylabel('Frequency of Measurement', fontsize=25, fontweight='bold')
plt.title('Solution State Distribution', fontsize=28, pad=15, fontweight = 'bold')

plt.xticks(sorted_keys, fontsize=12, fontweight='bold', rotation=45)
plt.yticks(fontsize=19)


plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.ylim(0, max(counts) * 1.18)   #
plt.tight_layout()
#plot_histogram(countsFinal, figsize = (10,10), fontsize = 20)
#plt.hist(countsFinal, 16)
plt.show()
nx.draw(thisG, with_labels = True)
plt.show()

#get expectation values
for i in range(0,n):
   zi = zExpect(countsFinal, i)
   print(f"Node {i}: <Z{i}> = {zi:.4f}")

for edge in thisG.edges():
   i,j = edge
   zizj = zzExpect(countsFinal, i, j)
   print(f"Edge ({i},{j}): ⟨Z{i}Z{j}⟩ ≈ {zizj:.4f}")
   cut = (1-zizj) / 2
   print(f"  → Prob this edge is cut = {cut:.4f}\n")
