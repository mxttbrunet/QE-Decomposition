import networkx as nx
import random
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('TkAgg')
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
import time

custom = """
1 2
2 3
2 4
3 5
4 5
4 6
5 8
6 7
7 8
7 9
8 10
9 10
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




cn = 8


reps = 1
shots = 500
M = 7
tau = 0.2

def getGString(g):
   for u,v in g.edges():
      print(f"{u + 1} {v + 1}")

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


def kExps(graph, K, V2, tran):
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

   kv = {}
   for ki in newK:
      for vj in newV2:
         kv[(ki, vj)] = zzExpect(counts, ki, vj)


   kDexp = {}
   kDexpNew = {}
   for a in range(len(newK)):
      for b in range(a + 1, len(newK)):
         ka,kb = newK[a], newK[b]
         ni, nj = from_idx[ka], from_idx[kb]
         kDexpNew[(ni,nj)] = zzExpect(counts,ka,kb)
               
   """
   for i in range(len(newK)):
      for j in range(i + 1, len(newK)):
         ki, kj = newK[i], newK[j]
         ni, nj = from_idx[ki], from_idx[kj]
         common = [v for v in newV2 if g_idx.has_edge(ki, v) and g_idx.has_edge(kj, v)]
         if common:
            kDexp[(ni, nj)] = sum(kv[(ki, v)] * kv[(kj, v)] for v in common) / len(common)
         else:
            kDexp[(ni, nj)] = 0.0
         print(f"===Base vs Tran===\n")
         print(f"Base {(ni,nj)} : {kDexpNew[(ni,nj)]}")
         print(f"Tran: {(ni,nj)} : {kDexp[(ni,nj)]}\n")
   """

   vDexp = {}
   for i in range(len(newV2)):
      for j in range(i + 1, len(newV2)):
         ni, nj = from_idx[newV2[i]], from_idx[newV2[j]]
         vDexp[(ni, nj)] = zzExpect(counts, newV2[i], newV2[j])

   return kSexp, kDexpNew, vSexp, vDexp



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
   #if(len(NKs) > 0):
   #   return {term[0]:term[1] for term in polyMap.items() if term[0] not in NKs}
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

   K = []
   numMerges = 0
   totalMerged = 0
   gaps = []
   gNum = -1
   round = 1
   cg = g.copy()
   cIsh = 0
   currCost = toClassical(g, [])
   fixTable = {}
   while(len(cg.nodes()) > limit):
      V2andK, K, V1, V2 = getInduced(cg)
      if(round % 10 == 0):
         print(f"Round {round}")
      noMergeSignal = False
      #draw(V2andK)
      if(len(K) > 3):
         kOld = len(K)
         mergedN, cg, K, _ = mergeAndUpdate2(cg, V2andK, K, V2, gNum)
         V2andK = cg.subgraph(K+V2)
         gaps.append(kOld - len(K))
         #draw(cg)
         #print(f"mergedN: {mergedN}")
         gNum -= len(mergedN)
         tableUpdate(mergedN, fixTable)
         #print(f"fixTable:{fixTable}")
         noMergeSignal = not mergedN
      if(len(K) > 7):
         break
      if(len(K) > 3 and noMergeSignal):
         J_list = gurobiReweight(V2andK, K, V2)
      else:
         J_list = ReWeight(V2andK,K,V2)
      #print(J_list)
      cg.remove_nodes_from(V2)
      for edge,val in list(J_list.items()):
         if(edge == 'seaHat'):
            cIsh+= J_list['seaHat']
         elif((edge[0],edge[1]) in cg.edges()):
            cg[edge[0]][edge[1]]['weight'] += val
         else:
            cg.add_edge(edge[0],edge[1])
            cg[edge[0]][edge[1]]['weight'] = val
      #draw(cg)
      round+=1
   
   #draw(cg)
   #print(f"funny maxCUT of this final graph:{nx.approximation.randomized_partitioning(cg, seed=1)}")
   if(len(K) >= 7):
      bestCut = solvePartial("full", cg, [])
   else:
      bestCut, bestSet = bruteMaxCut(cg)
   print(f"Final Round: {round - 1}")

   print(f"gaps:{gaps}")
   print(f"!!!seaHat:{cIsh}\n!!!table:{fixTable}")
   print(f"Best cut on final is {bestCut}\n")
   crowd = set()
   for key, val in fixTable.items():
      crowd.update(set(val)) 
   return cIsh + bestCut, crowd, -1*(gNum + 1), gaps




def mergeAndUpdate2(g0, sub0, K, V2, gStart):
   kSingle, kDouble, vSingle, vDouble = kExps(sub0, K, V2, True)
   #print(f"K double: {kDouble}\n")
   gSets = [set(pair[0]) for pair in kDouble.items() if pair[1] >= tau]
   groups = group(gSets)
   labeling = {}
   internalW = 0
   if(len(groups) >= 1):
      # Label every group up front and map each member -> its supernode, so a
      # neighbor can still be resolved to the right supernode even after its
      # group-mates have already been removed from g0.
      superOf = {}
      for gr in groups:
         labeling[gStart] = gr
         for node in gr:
            superOf[node] = gStart
         K.append(gStart)
         gStart -= 1

      toAdd = {}
      toRemove = []
      for sup, gr in labeling.items():
         g0.add_node(sup)   # keep the supernode even if it has no external edges
         for node in gr:
            for nbr in g0[node]:
               # Remap the neighbor to its supernode if it is also being merged;
               # otherwise it stays a plain (or already-merged) node.
               nbrSup = superOf.get(nbr, nbr)
               if(nbrSup == sup):
                  # internal edge: nodes are removed as we go, so each undirected
                  # internal edge is seen exactly once here.
                  internalW += g0[node][nbr]['weight']
               elif((sup, nbrSup) in toAdd):
                  toAdd[(sup, nbrSup)] += g0[node][nbr]['weight']
               else:
                  toAdd[(sup, nbrSup)] = g0[node][nbr]['weight']
               toRemove.append((node, nbr))
            g0.remove_edges_from(toRemove)
            g0.remove_node(node)
            K.remove(node)
            toRemove = []

      g0.add_weighted_edges_from([(u, v, w) for (u, v), w in toAdd.items()])

   #draw(g0)
   #print(f"INTERNAL: {internalW}")
   return labeling, g0, K, internalW

def genPerms(num):
   b = []
   for i in range(2**num):
      s = bin(i)[2:]
      s = "0" * (num-len(s)) + s
      yield s

def solvePartial(pos, subG, K):
   m = gp.Model("pSolver")
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
   
   interior = 0
   p = 0
   if(pos != "full"):
      for k in K:
         m.addConstr(x[k] == pos[p])
         p+=1

      
   m.update()
   m.optimize()
   obj = m.getObjective()
   obj_val = obj.getValue()

   """if(pos == "full"):
      side_0 = set()
      side_1 = set()
      for i in subG.nodes:
         val = round(x[i].X)  # .X gives the solved value; round handles float noise
         if val == 1:
            side_1.add(i)
         else:
            side_0.add(i)

   # --- Retrieve the cut edges themselves ---
      cut_edges = [j for j in subG.edges if round(y[j].X) == 1]

   # --- Summary ---
      cut_value = obj.getValue() 
      print(f"Optimal cut value: {cut_value}") 
      print(f"Side 0 ({len(side_0)} nodes): {sorted(side_0)}")
      print(f"Side 1 ({len(side_1)} nodes): {sorted(side_1)}")
   """
   m.update()
   m.close()
   #m.display()
   """with open("model.lp", "w") as f:
      m.write("model.lp")

   with open("model.lp", "r") as f:
       print(f.read())
   """
   return obj_val



def ReWeight(sub, K, V2):
   # double-count every K-K edge.  Strip them before solving.
   sub = sub.copy()
   kM = len(K)
   Kset = set(K)
   init = toClassical(sub, [])
   kInts = []
   for kNode in K:
      kInts.append( ("x" + str(kNode), init["x" + str(kNode)]) )
   interiors = [(u,v,w) for (u, v, w) in sub.edges.data("weight") if u in Kset and v in Kset ]
 
   #print(f"kInts:{kInts}")
   #print(f"Interiors:{interiors}")

   sub.remove_edges_from([(u, v) for u, v in sub.edges() if u in Kset and v in Kset])
   oneIdx = 0   
   c_hat = 0
   b = []
   gen = list(genPerms(kM))
   if(kM <= 20):
      additive = 0
      slvRows = []
      fvect = []
      rows = [bin for bin in gen if bin.count('1') <= 1]
      for l in range (kM + 1):
         currBin = rows[l]
         theString =[int(currBin[i]) for i in range(kM)]
         if(l>0):
            oneIdx = theString.index(1)
            additive = kInts[oneIdx][1]
            #print(f"additive:{additive}")

            slvRow = [(x + 1) % 2 for x in theString]
         else:
            slvRow = [x for x in theString]
         b.append(theString)
         partialSol = solvePartial(theString,sub,K)
         b[-1].append(partialSol)
         
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


def gurobiReweight(sub, K, V2):
   sub = sub.copy()
   kM = len(K)
   Kset = set(K)
   sub.remove_edges_from([(u, v) for u, v in sub.edges() if u in Kset and v in Kset])

   posOf = {node: idx for idx, node in enumerate(K)}
   edgePairs = list(combinations(K, 2))
   assigns = [[int(bit) for bit in bits] for bits in genPerms(kM)]

   rhs = [solvePartial(a, sub, K) for a in assigns]
   seaHat = rhs[0]
   rhs = [v - seaHat for v in rhs]

   m = gp.Model("gurobiReweight")
   m.setParam("OutputFlag", 0)

   w = {pair: m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name=f"w{pair[0]}_{pair[1]}")
        for pair in edgePairs}
   e = {i: m.addVar(vtype=GRB.CONTINUOUS, name=f"e{i}") for i in range(len(assigns))}
   m.update()

   for i, a in enumerate(assigns):
      cutTerms = gp.quicksum(w[pair] for pair in edgePairs if a[posOf[pair[0]]] != a[posOf[pair[1]]])
      m.addConstr(e[i] + cutTerms == rhs[i])

   m.setObjective(gp.quicksum(e[i] for i in range(len(assigns))), GRB.MINIMIZE)
   m.update()
   m.optimize()

   results = {'seaHat': seaHat}
   for pair in edgePairs:
      results[pair] = w[pair].X
   m.close()
   return results


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
      graphStrings = {}
      numToDo = 20
      stats = [[0,0,0,0,0], [0,0,0,0,0]]
      sNames = ["Average APX : ","Average Total Merged : ","Average Merge Occurances : ", "Average of Average K reduction : ", "Average non-0 K reduction : "]
      for j in range(numToDo):
         print(f"Graph {j}:\n")
         testG = nx.random_regular_graph(3,400)
         for u,v in testG.edges():
            testG[u][v]['weight'] = 1
         #draw(testG)
         #graphStrings[j] = getGString(testG) 
         bestCut = solvePartial("full", testG, [])
         #val, part = nx.approximation.one_exchange(testG, seed = 1)
         #print(f"oneExch:{val} with {part}")
         for i in list([0.1,0.2]):
            global tau
            tau = i
            total, mergedNodes, numMerges, diffs = decompo(testG, M)
            print(f"\n\n ====FINAL RESULTS====\n\n")
            print(f"Tau Parameter : {i}\n")
            print(f"Approximate/ Actual :  {total} / {bestCut} : {total / bestCut}\n")
            print(f"Total Merged Nodes : {len(mergedNodes)}\n")
            print(f"Number of Total Merge Occurances  : {numMerges}\n")
            print(f"Average K reduction : {sum(diffs) / len(diffs)}")
            nonZero = [d for d in diffs if d > 0]
            stats[int(str(i)[-1]) - 1][0]+= (total / bestCut) / numToDo
            stats[int(str(i)[-1]) - 1][1]+= (len(mergedNodes)) / numToDo
            stats[int(str(i)[-1]) - 1][2]+= (numMerges) / numToDo
            stats[int(str(i)[-1]) - 1][3]+= (sum(diffs) / len(diffs)) / numToDo
            
            if(len(nonZero) > 0):
               print(f"Average non-0 K reduction : {sum(nonZero) /  len([d for d in diffs if d > 0])}")
               stats[int(str(i)[-1]) - 1][4]+= sum(nonZero) / len(nonZero)/20
      for t in stats:
         print(f"\n\n FINAL TAU = {(stats.index(t) + 1) / 10 } STATS OVER {numToDo} GRAPHS")
         for st in range(len(t)):
            print(sNames[st] + str(t[st]))
         print("==========\n")

if __name__ == "__main__":
   nG = nx.erdos_renyi_graph(30,0.1)
   while (not nx.is_connected(nG) ):
      nG = nx.erdos_renyi_graph(30,0.1)
   for u,v in nG.edges():
      nG[u][v]['weight'] = 1
   sT = time.time()
   total, mergedNodes, numMerges, diffs = decompo(nG, 7)
   eT = time.time()
   print(eT - sT)
   #startUp()
