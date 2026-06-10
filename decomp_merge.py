import networkx as nx
import random
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import random
from itertools import combinations
from qiskit.quantum_info import SparsePauliOp as spo
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit.transpiler import generate_preset_pass_manager
from qiskit import transpile as qk_transpile
import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import minimize
import qiskit_aer as Aer
from qaoa_helpers import *

custom = """
1 2
1 3
1 6
2 3
2 4
3 4
4 5
4 6
5 6
"""



"""
1 2
1 3
1 4
2 5
2 6
3 5
3 6
4 5
4 6
"""

cn = 6
reps = 1
shots = 1600
M = 5
tau = 0.25

def dec(strang):
   temp = int(strang)
   temp-=1
   return str(temp)

def group(pairs):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        parent[find(a)] = find(b)

    for pair in pairs:
        items = list(pair)
        for item in items[1:]:
            union(items[0], item)

    groups = {}
    for x in parent:
        groups.setdefault(find(x), set()).add(x)

    return list(groups.values())

def draw(g): 
   pos = nx.spring_layout(g)
   nx.draw(g, pos, with_labels = True) 
   edge_labels = nx.get_edge_attributes(g, "weight")
   nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels)
   plt.show()

def build_pauli_list(graph): 
   pauli_list = []
   offset = 0.0 
   nodes = list(graph.nodes())

   for u, v, data in graph.edges(data=True):
      w = data.get('weight', 1.0)
      pauli_list.append(('ZZ', [u, v], -w / 2.0))
      offset += w / 2.0


   return pauli_list, offset


def kExps(graph, K, V2):
   nodes = list(graph.nodes())
   n = len(nodes)

   to_idx   = {v: i for i, v in enumerate(nodes)}
   from_idx = {i: v for v, i in to_idx.items()}
   newK = [to_idx[k] for k in K]
   newV2 =[to_idx[v] for v in V2]
   g_idx    = nx.relabel_nodes(graph, to_idx, copy = True)

   lf_idx = {}

   pauli_list, _offset = build_pauli_list(g_idx)

   costH      = spo.from_sparse_list(pauli_list, num_qubits=n)
   result, transpiled_ansatz, objVals = optimize_qaoa(costH, reps = reps)

   backend  = Aer.AerSimulator()
   opt_circ = transpiled_ansatz.assign_parameters(result.x)
   opt_circ.measure_all()
   sim_circ = qk_transpile(opt_circ, backend=backend, optimization_level=0)
   counts   = backend.run(sim_circ, shots=shots).result().get_counts()

   kSexp  = {from_idx[k]: zExpect(counts, k) for k in newK}
   vSexp  = {from_idx[v]: zExpect(counts, v) for v in newV2}
   kDexp = {}
   vDexp = {}
   for i in range(len(newK)):
      for j in range(i + 1, len(newK)):
         ni, nj = from_idx[newK[i]], from_idx[newK[j]]
         kDexp[(ni, nj)] = zzExpect(counts, newK[i], newK[j])

   for i in range(len(newV2)):
      for j in range(i + 1, len(newV2)):
         ni, nj = from_idx[newV2[i]], from_idx[newV2[j]]
         vDexp[(ni, nj)] = zzExpect(counts, newV2[i], newV2[j])
    
   return kSexp, kDexp, vSexp, vDexp



def makeCustom(stringy, numN):
   newg = nx.Graph()
   newg.add_nodes_from(list(range(0, numN)))
   tokens = stringy.split()
   edgesN = []
   for i in range(0, len(tokens) - 1, 2):
      u = int(tokens[i]) - 1
      v = int(tokens[i + 1]) - 1
      edgesN.append((u, v, 1))
   newg.add_weighted_edges_from(edgesN)
   return newg


def toClassical(graph, Ks):
   polyMap = {}
   NKs = ["x" + str(k) for k in Ks]
   for u,v in graph.edges():
      uT = "x" + str(u) 
      vT = "x" + str(v) 
      w = graph[u][v]['weight']

      if (uT) not in polyMap: 
         polyMap[uT] = w 
      else:
         polyMap[uT]+=w
      if (vT) not in polyMap:
         polyMap[vT] = w

      else:
         polyMap[vT]+= w
      if(uT + vT) not in polyMap:
         polyMap[uT+vT]= -2*w
      else:
         polyMap[uT+vT]+=  -2*w
   fullC = ""
   for entry in polyMap.items():
      if(entry[0] in NKs):
         pass
      elif(entry[1] > 0):
         fullC+= (f" +{entry[1]}{entry[0]}")
      elif(entry[1] < 0):
         fullC+= (f" {entry[1]}{entry[0]}")
      else:
         pass
   #print("man made: " + fullC)
   if(len(NKs) > 0):
      return {term[0]:term[1] for term in polyMap.items() if term[0] not in NKs}
   else:
      return polyMap

def toIsing(polyM):
   
   polyH = {}
   polyH["I"] = 0
   for term in polyM.items():
      if(term[0].count('x') == 1):
         zU = "Z" + term[0][1:]
         polyH["I"] += term[1] / 2
         
         if zU not in polyH:
            polyH[zU] = -1 * term[1]  / 2
         else:
            poluH[zU] += -1* term[1] / 2
      elif(term[0].count('x') == 2):
          polyH["I"]+= term[1] / 4
          ts = term[0].split("x")
          polyH[("Z" + ts[1])] += -1 * term[1] / 4
          polyH[("Z" + ts[-1])] += -1 * term[1] / 4
          zD = "Z"+ts[1] + "Z" + ts[-1]
          if zD not in polyH:
             polyH[zD] = term[1] / 4
          else:
             polyH[zD] += term[1] / 4
   fullH = ""
   for entry in polyH.items():
      if(entry[1] > 0):
         fullH+= (f" +{entry[1]}{entry[0]}")
      elif(entry[1] < 0):
         fullH+= (f" {entry[1]}{entry[0]}")
      else:
         pass

   print(fullH)
   return polyH

def getInduced(graph):
   K = list(nx.minimum_node_cut(graph))
   #TESTING K
   #K = [1,2,3]
   GslashK = graph.copy()
   GslashK.remove_nodes_from(K)
   fragments = sorted(nx.connected_components(GslashK), key=len)
   V2 = list(fragments[0])
   V1 = [v for comp in fragments[1:] for v in comp]
   GslashV1 = graph.copy()
   GslashV1.remove_nodes_from(V1)
   return GslashV1, K, V1, V2

def tableUpdate(newM, fixTable):
   for key,val in newM.items():
      poor = [node for node in val if int(node) < 0]
      fixTable[key] = set(newM[key])
      for neg in poor:
         fixTable[key].update(fixTable[neg])
         fixTable[key].remove(neg)
         fixTable.pop(neg)
   
   return fixTable


def decompo(g, limit):
   gNum = -1
   round = 1
   cg = g.copy()
   cIsh = 0
   currCost = toClassical(g, [])
   fixTable = {}
   while(len(cg.nodes()) > limit):
      V2andK, K, V1, V2 = getInduced(cg)
      print(f"---Round {round}: K={K}, V1={V1}, V2={V2}---")
      if(len(K) > 3):
         #draw(V2andK)
         mergedN, cg, K = mergeAndUpdate(cg, V2andK, K, V2, gNum)
         V2andK = cg.subgraph(K+V2)
         #draw(cg)
         print(f"mergedN: {mergedN}")
         gNum -= len(mergedN)
         tableUpdate(mergedN, fixTable)
         print(f"fixTable:{fixTable}")
      J_list = ReWeight(V2andK,K,V2)
      #print(J_list)
      cg.remove_nodes_from(V2)
      for edge,val in J_list.items():
         if(edge == 'seaHat'):
            cIsh+= J_list['seaHat']
         elif((edge[0],edge[1]) in cg.edges()):
            cg[edge[0]][edge[1]]['weight'] = val
         else:
            cg.add_edge(edge[0],edge[1])
            cg[edge[0]][edge[1]]['weight'] = val
      #draw(cg)
      round+=1
   print(f"!!!seaHat:{cIsh}\n!!!table:{fixTable}")
   draw(cg)
   #print(f"funny maxCUT of this final graph:{nx.approximation.randomized_partitioning(cg, seed=1)}")
   bestCut, bestSet = bruteMaxCut(cg)
   print(f"Best cut on final is {bestCut} on {bestSet}\n")
   exit()




def mergeAndUpdate(g0, sub0, K, V2, gStart):
   kSingle, kDouble, vSingle, vDouble = kExps(sub0, K, V2)
   print(f"K single: {kSingle}\nK double: {kDouble}\nV2 single: {vSingle}\nV2 double:{vDouble}")
   ##maybe look into merging the V1s? 
   gSets = [set(pair[0]) for pair in kDouble.items() if pair[1] >= tau]
   groups = group(gSets)
   labeling = {}
   toRemove = []
   toAdd = {}
   for i in range(len(groups)):
      #print(f"Group {gStart}: {groups[i]}")
      labeling[gStart] = set([node for node in groups[i]])
      g0.add_node(gStart)
      K.append(gStart)
      for node in groups[i]:
         for nbr, dic in g0.adj[node].items():
            if( (gStart,nbr) in toAdd ):
               toAdd[(gStart,nbr)]+= g0[node][nbr]['weight']
            else:
               toAdd[(gStart,nbr)] = 1.0

            toRemove.append((node,nbr))
      
      gStart-=1
   g0.remove_edges_from(toRemove)
   g0.add_weighted_edges_from([(u,v,w) for (u,v),w in toAdd.items() ])
   g0.remove_nodes_from([x for x in g0.nodes() if g0.degree[x] == 0])
   newK = []
   for new in K:
      if(new in g0):
         newK.append(new)
   return labeling, g0, newK

def genPerms(num):
   b = []
   for i in range(2**num):
      s = bin(i)[2:]
      s = "0" * (num-len(s)) + s
      yield s

def solvePartial(pos, subG, K):
   m = gp.Model("pSolver")
   cl = toClassical(subG, K)
   m.ModelSense = GRB.MAXIMIZE
   m.update()
   m.setParam("OutputFlag",0) 
      
   x = {}  # vertex variables
   y = {}  # edge variables

   for i in list(subG.nodes):
      x[i] = m.addVar(vtype=GRB.BINARY, name='x' + str(i))

   for j in list(subG.edges):
      y[j] = m.addVar(vtype=GRB.BINARY, obj = subG[j[0]][j[1]]['weight'], name= 'y'  + str(j))
   m.update()

   for j in list(subG.edges): #j is a tuple since it is an element in the set of edges
      m.addConstr(y[j] <= x[j[0]] + x[j[1]]) #binary 
      m.addConstr(y[j] <= 2 - x[j[0]] +- x[j[1]]) #color of edge - if both vertices are 1, then I can't do it
      m.addConstr(y[j] >= x[j[0]] - x[j[1]]) 
      m.addConstr(y[j] >= x[j[1]] - x[j[0]]) 

   p = 0
   for k in K:
      m.addConstr(x[k] == pos[p])
      p+=1
      
   m.update()
   m.optimize()
   obj = m.getObjective()
   obj_val = obj.getValue()
   m.update()
   m.display()
   """with open("model.lp", "w") as f:
      m.write("model.lp")

   with open("model.lp", "r") as f:
       print(f.read())
   """
   return obj_val

def ReWeight(sub, K, V2):
   c_hat = 0
   kM = len(K)
   b = []
   gen = list(genPerms(kM))
   if(kM <= 20):
      slvRows = []
      fvect = []
      rows = [bin for bin in gen if bin.count('1') <= 1]
      for l in range (kM + 1):
         currBin = rows[l]
         theString =[int(currBin[i]) for i in range(kM)]
         if(l>0):
            slvRow = [(x + 1) % 2 for x in theString]
         else:
            slvRow = [x for x in theString]
         b.append(theString)
         b[-1].append(solvePartial(theString,sub,K))
         if(kM == 1):
            return dict([('seaHat', b[-1][-1])])
         fvect.append(b[-1][-1])
         b[-1].insert(0, 1)
         slvRow.insert(0,1)
         slvRows.append(slvRow)

      coeffMtx = np.array(slvRows)
      fvectArr = np.array(fvect)
      #print(b)
            #print(f"coeff: {coeffMtx}\n bVec:{fvectArr}")
      x = np.linalg.solve(coeffMtx,fvectArr)
      #print(f"solution: {x}")
      edgePairs = list(combinations(K,2))
      un = ['seaHat'] + edgePairs
      results = dict(zip(un,x))
      return results
   else:
      print("too many k right now :(")
      print(f"K:{K}")
      exit()
      #nextP = next(gen)
      #bitS = [int(nextP[j]) for j in range(len(k))]
   
   return b


   






def bruteMaxCut(graph):
   nodes = list(graph.nodes())
   best = 0
   bestSet = None
   for assign in genPerms(len(nodes)):
      side = {nodes[i]: int(assign[i]) for i in range(len(nodes))}
      cut = 0
      for u, v, data in graph.edges(data=True):
         if side[u] != side[v]:
            cut += data.get('weight', 1.0)
      if cut > best:
         best = cut
         bestSet = side
   return best, bestSet

def startUp():
   mode = input("Select operation: Measure mode (0)...Graph mode (1)? \n")
   if(mode == "0"):
      pList = []
      off = 0
      qbits = set()
      print("Entering measure mode. Type ! when done.")
      while (True):
         term = input("Ready for the next term, format 'coeffZiZj...Zf':")
         if(term == "!"):
            break
         qs = term.split("Z")
         coef = float(qs.pop(0))
         qbits.update([int(x) for x in qs])
         pList.append(('Z'*(len(qs)), [int(x) for x in qs], coef))
      n = len(qbits)
      costH = spo.from_sparse_list(pList, num_qubits=n)
      result, transpiled_ansatz, objective_func_vals = optimize_qaoa(costH, reps=reps)
      backend  = Aer.AerSimulator()
      opt_circ = transpiled_ansatz.assign_parameters(result.x)
      opt_circ.measure_all()
      countsF = backend.run(opt_circ, shots=shots).result().get_counts()
   
      sExp = {}
      dExp = {}
      #print(countsF)
      for i in range(0,n):
         sExp["Z" + str(i)] = zExpect(countsF, i)
      for m in range (0,n):
         for l in range(m+1,n):
            dExp["Z" + str(m) + "Z" + str(l)] =zzExpect(countsF, m,l)
      print(f"singles: {sExp}")
      print(f"doubles: {dExp}")

   elif(mode == "1"):
      #testG = makeCustom(custom, cn)
      testG = nx.random_regular_graph(3, 20)
      for u,v in testG.edges():
         testG[u][v]['weight'] = 1
      draw(testG)
      print("")
      bestCut, bestSet = bruteMaxCut(testG)
      print(f"best maxCut {bestCut} on {bestSet}\n")
      decompo(testG, M)


if __name__ == "__main__":
   startUp()
