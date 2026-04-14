import itertools
import gurobipy as gp
import networkx as nx
import matplotlib.pyplot as plt
from gurobipy import GRB

from qaoa_helpers import (
    buildPaulis, zExpect, zzExpect,
    cost_func, optimize_qaoa, run_basic_qaoa,
    genGraph, makeCustom,
)


def _solve_maxcut_fixed(subgraph, fixed_vals):
    nodes      = list(subgraph.nodes())
    free_nodes = [v for v in nodes if v not in fixed_vals]

    if not free_nodes:
        val = 0.0
        for u, v in subgraph.edges():
            w  = subgraph[u][v].get('weight', 1.0)
            ci = float(fixed_vals[u])
            cj = float(fixed_vals[v])
            val += w * (ci + cj - 2.0 * ci * cj)
        return val

    model = gp.Model()
    model.setParam("OutputFlag", 0)

    z = {}
    for v in nodes:
        if v in fixed_vals:
            z[v] = float(fixed_vals[v])
        else:
            z[v] = model.addVar(vtype=GRB.BINARY, name=f"z_{v}")

    obj = gp.LinExpr()
    for u, v in subgraph.edges():
        w  = subgraph[u][v].get('weight', 1.0)
        zi = z[u]
        zj = z[v]
        zi_const = isinstance(zi, float)
        zj_const = isinstance(zj, float)

        if zi_const and zj_const:
            obj.addConstant(w * (zi + zj - 2.0 * zi * zj))
        elif zi_const:
            obj.addConstant(w * zi)
            obj += w * (1.0 - 2.0 * zi) * zj
        elif zj_const:
            obj.addConstant(w * zj)
            obj += w * (1.0 - 2.0 * zj) * zi
        else:
            # Linearise z_i XOR z_j with a binary auxiliary y
            y = model.addVar(vtype=GRB.BINARY)
            model.addConstr(y >= zi - zj)
            model.addConstr(y >= zj - zi)
            model.addConstr(y <= zi + zj)
            model.addConstr(y <= 2.0 - zi - zj)
            obj += w * y

        model.update()

    model.setObjective(obj, GRB.MAXIMIZE)
    model.optimize()
    model.update()
    if model.status == GRB.OPTIMAL:
        return model.objVal
    else:
        print(f"  [reweight] Gurobi status {model.status} — returning 0")
        return 0.0


def reweight(K, graph, V2):
    """Algorithm 2 (Reweight) from Ponce et al. 2025."""
    K_list = list(K)
    k      = len(K_list)

    sub_nodes = K_list + list(V2)
    H = graph.subgraph(sub_nodes).copy()
    for u, v in H.edges():
        H[u][v].setdefault('weight', 1.0)

    all_s = list(itertools.product([0, 1], repeat=k))
    b = []
    for s in all_s:
        fixed = {K_list[i]: s[i] for i in range(k)}
        cs    = _solve_maxcut_fixed(H, fixed)
        b.append(cs)
        #print(f"  s={s}  C_s={cs:.4f}")

    lp = gp.Model()
    lp.setParam("OutputFlag", 0)

    J_edge = {}
    for i in range(k):
        for j in range(i + 1, k):
            J_edge[(K_list[i], K_list[j])] = lp.addVar(
                lb=-GRB.INFINITY, name=f"Je_{K_list[i]}_{K_list[j]}"
            )

    J_diag = {
        K_list[i]: lp.addVar(lb=-GRB.INFINITY, name=f"Jd_{K_list[i]}")
        for i in range(k)
    }

    c_hat_var = lp.addVar(lb=-GRB.INFINITY, name="c_hat")
    e_vars    = [lp.addVar(lb=0.0, name=f"e_{ind}") for ind in range(len(all_s))]

    for ind, s in enumerate(all_s):
        active = [K_list[i] for i in range(k) if s[i] == 1]
        lhs    = c_hat_var + e_vars[ind]
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                u, v = active[i], active[j]
                key  = (u, v) if (u, v) in J_edge else (v, u)
                lhs  = lhs + J_edge[key]
        for u in active:
            lhs = lhs + J_diag[u]
        lp.addConstr(lhs == b[ind])

    lp.setObjective(gp.quicksum(e_vars), GRB.MINIMIZE)
    lp.optimize()

    if lp.status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        print(f"  [reweight] LP status {lp.status} — returning zero weights")
        return {}, 0.0

    total_error = lp.ObjVal
    if total_error > 1e-6:
        print(f"  [reweight] LP residual error: {total_error:.6f}  (|K|={k} > 3)")

    J_hat = {}
    for (u, v), var in J_edge.items():
        J_hat[(u, v)] = var.X
    for u, var in J_diag.items():
        J_hat[(u, u)] = var.X
    c_val = c_hat_var.X

    return J_hat, c_val


def _print_objective(G, c_total):
    """Print the current QUBO objective function for the reduced graph."""
    terms = [f"{c_total:+.4f}"]
    for v in sorted(G.nodes()):
        h = G.nodes[v].get('weight', 0.0)
        if abs(h) > 1e-12:
            terms.append(f"{h:+.4f}·z{v}")
    for u, v in sorted(G.edges()):
        w = G[u][v].get('weight', 1.0)
        if abs(w) > 1e-12:
            terms.append(f"{w:+.4f}·z{u}·z{v}")
    print("  Objective: C(z) = " + " ".join(terms))


def decomp(G_input, M=4):
    """Algorithm 1 (Decomp) from Ponce et al. 2025."""
    G = G_input.copy()
    for u, v in G.edges():
        G[u][v].setdefault('weight', 1.0)
    for v in G.nodes():
        G.nodes[v].setdefault('weight', 0.0)

    c_total   = 0.0
    iteration = 0

    while True:
        n = G.number_of_nodes()

        if G.number_of_edges() >= n * (n - 1) // 2 and n > 1:
            print(f"[decomp] iter {iteration}: graph is complete ({n} nodes). Done.")
            break

        try:
            K = list(nx.minimum_node_cut(G))
        except nx.NetworkXError:
            print("[decomp] Cannot compute min-cut (disconnected?). Done.")
            break

        if not K:
            print("[decomp] No vertex cut found (complete or trivial graph). Done.")
            break

        if len(K) >= M:
            print(f"[decomp] iter {iteration}: |K|={len(K)} >= M={M}. Done.")
            break

        G_minus_K  = G.copy()
        G_minus_K.remove_nodes_from(K)
        components = sorted(nx.connected_components(G_minus_K), key=len)

        if len(components) < 2:
            print("[decomp] K did not disconnect graph. Done.")
            break

        V2 = list(components[0])
        V1 = [v for comp in components[1:] for v in comp]

        print(f"[decomp] iter {iteration}: n={n}, |K|={len(K)}, "
              f"|V1|={len(V1)}, |V2|={len(V2)}\n V1:{V1}  K:{K}   V2:{V2}")

        J_hat, c_hat = reweight(K, G, V2)
        print(f"J_HAT:{J_hat}, c_HAT:{c_hat}")

        new_nodes = V1 + list(K)
        G_new = nx.Graph()
        G_new.add_nodes_from(new_nodes)

        for v in new_nodes:
            G_new.nodes[v]['weight'] = G.nodes[v].get('weight', 0.0)

        for u in new_nodes:
            for v in new_nodes:
                if u < v and G.has_edge(u, v):
                    G_new.add_edge(u, v, weight=G[u][v]['weight'])

        for key, w in J_hat.items():
            if key[0] == key[1]:
                node = key[0]
                if node in G_new:
                    G_new.nodes[node]['weight'] += w
            else:
                u, v = key
                if u in G_new and v in G_new:
                    if G_new.has_edge(u, v):
                        G_new[u][v]['weight'] += w
                    else:
                        G_new.add_edge(u, v, weight=w)

        c_total += c_hat
        G        = G_new
        iteration += 1

        # Print new objective function
        _print_objective(G, c_total)

        # Draw updated reweighted graph
        pos = nx.spring_layout(G_new, seed=42)
        fig, ax = plt.subplots()
        nx.draw(G_new, pos,
            ax=ax,
            with_labels=True,
            node_color='lightblue',
            node_size=700,
            font_size=12,
            font_weight='bold',
            edge_color='gray',
            width=2)
        edge_labels = {e: f"{w:.3g}" for e, w in nx.get_edge_attributes(G_new, 'weight').items()}
        nx.draw_networkx_edge_labels(G_new, pos,
                                     edge_labels=edge_labels,
                                     ax=ax,
                                     font_color='red',
                                     font_size=10,
                                     font_weight='bold')
        ax.set_title(f"After iteration {iteration}  (c_total={c_total:.4f})")
        plt.tight_layout()
        plt.show()

    return G, c_total


def reCut(graph, M=4, reps=1, shots=40000):
    """Full pipeline: decompose graph via Algorithm 1, then run QAOA on the reduced graph."""
    for u, v in graph.edges():
        graph[u][v].setdefault('weight', 1.0)

    n_orig = graph.number_of_nodes()
    print(f"\n=== reCut: {n_orig} nodes, {graph.number_of_edges()} edges, M={M} ===")

    G_reduced, c_total = decomp(graph, M=M)

    n_r = G_reduced.number_of_nodes()

    if n_r == 0:
        print("[reCut] Empty graph after decomposition.")
        return {}, {}, c_total, G_reduced

    nx.draw(G_reduced, with_labels=True)
    plt.title(f"Reduced Graph ({n_r} nodes)")
    plt.show()

    singExp, doubExp = run_basic_qaoa(
        G_reduced, reps=reps, problem="maxcut",
        shots=shots, filter_z2=True, draw_graph=False
    )

    return singExp, doubExp, c_total, G_reduced


if __name__ == "__main__":
    n = 5

    custom = """
    1 2
    1 3
    2 3
    2 4
    3 5
    4 5
    """

    testG = makeCustom(custom, n)

    for u, v in testG.edges():
        testG[u][v]['weight'] = 1

    nx.draw(testG, with_labels=True)
    plt.show()

    sE, dE, offset, G_red = reCut(testG, M=4, reps=1, shots=40000)

    print(f"\nSingle Z expectations: {sE}")
    print(f"Double ZZ expectations: {dE}")
    print(f"Constant offset from decomp: {offset:.4f}")

    CS, part = nx.approximation.one_exchange(testG, seed=1)
    print(f"\nClassical approx (one_exchange): {CS}, partition={part}")
