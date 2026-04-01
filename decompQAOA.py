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
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from scipy.optimize import minimize
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.primitives import Estimator
import qiskit_aer as Aer

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


#Generate polynomial for system of Eq's 
def genPolyCut(graph, theC, kind):
   polyDict = {}
   if(kind):
      for (i,j) in graph.edges():
         if ( ("z" + str(i)) not in polyDict):
            polyDict["z" + str(i)] = 1
         else:
            polyDict["z" + str(i)]+=1

         if ( ("z" + str(j)) not in polyDict):
            polyDict["z" + str(j)] = 1
         else:
            polyDict["z" + str(j)]+=1

         polyDict["z" + str(i) + "z" + str(j)] = -2



   else:
      for (i,j) in graph.edges():
         if( (i not in theC) and (j not in theC) ):
            if(("z"+str(i)) not in polyDict):
               polyDict["z" + str(i)] = 1 
            else:
               polyDict["z" + str(i)]+=1
            if ( ("z" + str(j)) not in polyDict):
               polyDict["z" + str(j)] = 1
            else:
               polyDict["z" + str(j)]+=1
            polyDict["z" + str(i) + "z" + str(j)] = -2
         elif( (i in theC) and (j not in theC) ):
            if ( ("z" + str(j)) not in polyDict):
               polyDict["z" + str(j)] = 1
            else:
               polyDict["z" + str(j)]+=1
            polyDict["z" + str(i) + "z" + str(j)] = -2
         
         elif((i not in theC) and (j in theC) ):
            if ( ("z" + str(i)) not in polyDict):
               polyDict["z" + str(i)] = 1
            else:
               polyDict["z" + str(i)]+=1
            polyDict["z" + str(i) + "z" + str(j)] = -2

   return polyDict


#calculate single expectation vals
def zExpect(counts, i):
   totalZ = 0.0
   shots = sum(counts.values())
   for bitstring, count in counts.items():
       z_i = 1 if bitstring[-1-i] == '0' else -1
       #z_j = 1 if bitstring[-1-j] == '0' else -1
       totalZ += count * z_i
   return totalZ / shots


#calculate double expectation vals
def zzExpect(counts, i,j):
   totalZZ = 0.0
   shots = sum(counts.values())
   for bitstring, count in counts.items():
       z_i = 1 if bitstring[-1-i] == '0' else -1
       z_j = 1 if bitstring[-1-j] == '0' else -1
       totalZZ += count * z_i * z_j
   return totalZZ / shots
   

#build Rz and Rzz for QAOA
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

#optimize for fixed F
def maxWFixed(bStr, poly, cut):
   m = gp.Model("f")
   m.setParam('OutputFlag',0)
   indv = []
   for entry in poly:
      if(entry.count('z') == 1):
         indv.append(entry)
      elif(entry.count('z') == 2):
         sec = entry[1:].find('z')
         if(entry[:sec+1] not in indv):
            indv.append(entry[:sec+1])
         if(entry[sec + 1:] not in indv):
            indv.append(entry[sec + 1:])
   print(sorted(indv))
   indv = sorted(indv)
   m.addVars(len(indv), vtype = GRB.BINARY, name = indv)
   m.update()
   i = 0
   mvars = {}
   objf = 0
   for var in m.getVars():      
      #print(f"{var.varName}")
      mvars[var.varName] = var
      if(int(var.varName[1:]) in cut):
         m.addConstr(var== int(bStr[i]))
         i+=1
         print(f"{var.varName} set as {bStr[i-1]}")
   for entry in poly:
      if(entry.count('z') == 2):
         sec = entry[1:].find('z')
         objf += mvars[entry[:sec+1]] * mvars[entry[sec+1:]] * poly[entry]
      elif(entry.count('z') == 1):
         objf += mvars[entry] * poly[entry]
   print(objf)
   m.setObjective(objf, GRB.MAXIMIZE)
   m.optimize()

   #for v in m.getVars():
   #   print('%s %g' % (v.VarName, v.X))
   return m.ObjVal

#reweight edges
def reWeight(cut, v2K, J):
   sols = []
   c = 0
   polyN = genPolyCut(v2K, cut, kind = False)
   print(polyN)
   for i in range(0, (2**(len(cut))) ):
      binAp = str(bin(i))[2:]
      n = binAp.zfill(len(cut))
      sols.append([n,maxWFixed(n, polyN, cut)])
      #binAr.append(n)
   print(f"SOLS : {sols}")   


#find minVertCutSet,  reweight
def decomp(tGraph, lim):
   cutSet = sorted(list(nx.minimum_node_cut(thisG)))
   #cutSet = [2,3,4]
   copyG = thisG.copy()
   copyG.remove_nodes_from(cutSet)
   seps = sorted(list(nx.connected_components(copyG)),key = len, reverse = True )
   sepsv1 = []
   sepsv2 = []
   i= 0
   print(seps)
   if(len(seps) >= 2):
      for m in range(math.floor( len(seps) / 2)):
         for item in seps[m]:
            sepsv1.append(item)
      i = math.floor(len(seps) / 2)
      while(i < len(seps)):
         for item in seps[i]:
            sepsv2.append(item)
         i+=1
   print(sepsv1)
   print(sepsv2)
   print(f"cutset: {cutSet}")
   v1 = thisG.copy()
   v1.remove_nodes_from(list(sepsv1) + cutSet)
   v2 = thisG.copy()
   v2.remove_nodes_from(list(sepsv2) + cutSet)

   v2K = thisG.copy()
   v2K.remove_nodes_from(list(sepsv1))

   v1K = thisG.copy()
   v1K.remove_nodes_from(list(sepsv2))
#   if(len(v1) >= len(v2)):

   reWeight(cutSet, v1K, 2)

 #  else:
#   reWeight(cutSet, v2K, 2)





#hardcoding for example 


n = 6
thisG = nx.Graph()

#thisG = genGraph(n, -1, "sparse")

#thisG.add_nodes_from(range(1,n+1))
#thisG.add_edges_from([(1,2),(1,4),(1,5),(2,3),(3,4),(4,5)])

#graph from paper
thisG.add_edges_from([(1,2),(1,3),(1,4),(2,5),(2,6),(3,5),(3,6),(4,5),(4,6)])

nx.draw(thisG, with_labels=True)
plt.show()

decomp(thisG, 2)

