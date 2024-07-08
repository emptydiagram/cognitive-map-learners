import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx

def make_edge_node_matrix(edge_indices, n_nodes):
    """
    edge_indices - dict{edge -> edge index}. edge is a pair (u, v) of node indices
    """
    G = jnp.zeros((len(edge_indices), n_nodes))
    for (u, v), edge_index in edge_indices.items():
        G = G.at[edge_index, u].set(1)
    return G

def recovery(x, D, theta):
    # D ~ h x n, dictionary of column vectors. assume hyperdimensional, so all the columns are pseudo-orthogonal
    # x ~ h x 1, column vectors
    # theta: positive scalar
    # returns: d_i if there d_i is a row vector such that its cosine similarity with x is > theta, else 0
    D_sim = D.T @ x
    matches = D_sim >= theta
    if jnp.any(matches):
        col = jnp.argmax(matches)
        return D[:, jnp.argmax(matches)], col
    else:
        return jnp.zeros_like(x), None

def winner_take_all(x):
    win_idx = jnp.argmax(x)
    return jnp.zeros_like(x).at[win_idx].set(x[win_idx])

def build_toh_adj_matrix():
    n_nodes = 27

    # (big, medium, small) ring indices = (0, 1, 2)
    # any subset of rings can go on any peg, and the ToH constraint means
    # that the ordering is unique
    # state [a, b, c] means ring 0 is on peg a, ring 1 on peg b, ring 2 on peg c
    # aka mapping (ring indices) -> peg number
    #
    init_state = [0, 0, 0]

    adj_matrix = jnp.zeros((n_nodes, n_nodes))

    node_indices = { tuple(init_state): 0 }
    node_idx = 1

    # BFS
    # to_visit entries are in (prev_state, state) format
    to_visit = [init_state]
    visited = set()
    while len(to_visit) > 0:
        curr_state = to_visit.pop(0)
        visited.add(tuple(curr_state))

        # for state [a, b, c], which states are adjacent?
        # a ring cannot be moved if there is some ring on top of it.
        # also, for each ring we can look at available moves:
        #  - ring 0: cant be placed on top of ring 1 or ring 2, so rule out pegs with one of those
        #  - ring 1: cant be placed on top of ring 2
        #  - ring 2: can be placed anywhere

        a, b, c = curr_state
        r0_peg_cands = set()
        r1_peg_cands = set()

        alters = []
        if a != b and a != c:
            r0_invalid_pegs = { b, c }
            alters.extend([(0, i) for i in range(3) if i not in r0_invalid_pegs and i != a])
        if b != c:
            r1_invalid_pegs = { c }
            alters.extend([(1, i) for i in range(3) if i not in r1_invalid_pegs and i != b])
        alters.extend([(2, i) for i in range(3) if i != c])

        # for each edge from current state:
        #   add edge to the adj matrix
        #   if node not visited and not in to_visit
        #     add to to_visit
        for alter_ring, alter_peg in alters:
            alter_state = curr_state.copy()
            alter_state[alter_ring] = alter_peg
            alter_state_tup = tuple(alter_state)

            if alter_state_tup in visited:
                continue

            alter_is_discovered = alter_state_tup in node_indices.keys()
            if not alter_is_discovered:
                node_indices[alter_state_tup] = node_idx
                node_idx += 1

            i = node_indices[tuple(curr_state)]
            j = node_indices[tuple(alter_state)]
            adj_matrix = adj_matrix.at[i, j].set(1)
            adj_matrix = adj_matrix.at[j, i].set(1)

            if not alter_is_discovered:
                to_visit.append(alter_state)

    return adj_matrix, node_indices




class McDonaldCML:
    def __init__(self, key, params, G):
        param_names = ['n_obs', 'n_act', 'emb_dim', 'Q_init_stddev', 'V_init_stddev', 'eta_q', 'eta_v']
        n_obs, n_act, emb_dim, Q_init_stddev, V_init_stddev, eta_q, eta_v = itemgetter(*param_names)(params)

        self.eta_q = eta_q
        self.eta_v = eta_v

        key, key1, key2 = jax.random.split(key, 3)

        self.S = jax.random.normal(key1, (emb_dim, n_obs)) * Q_init_stddev
        self.A = jax.random.normal(key2, (emb_dim, n_act)) * V_init_stddev
        self.G = G


    def step(self, s_curr, s_goal, s_goal_thresh):
        # If the target state s* was not
        # in S, then it would be pseudo-orthogonal to all node states, δ(s*,
        # S) < θ, and the CML returned a zeros vector (the MAP identity
        # element under addition); else if δ(s*, S) ≥ θ, then the raw s*
        # vector was used.
        s_goal_matches = (self.S.T @ s_goal >= s_goal_thresh).any()
        s_goal_proc = s_goal if s_goal_matches else jnp.zeros_like(s_goal)

        s_curr_rec, s_idx = recovery(s_curr, self.S, s_goal_thresh)

        if s_idx is None:
            return s_curr_rec

        s_diff = s_goal_proc - s_curr_rec

        u = jnp.linalg.pinv(self.A) @ s_diff
        g = self.G[:, s_idx]

        c = winner_take_all(u * g)

        s_pred = s_curr_rec + self.A @ c
        

        # WTA



if __name__ == '__main__':
    adj_matrix, node_indices = build_toh_adj_matrix()

    adj_matrix_np = jnp.asarray(adj_matrix)
    G = nx.from_numpy_array(adj_matrix_np, create_using=nx.DiGraph)
    labels = { node_idx : f'{s[0]}{s[1]}{s[2]}' for s, node_idx in node_indices.items() }
    nx.set_node_attributes(G, labels, 'label')
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw(G, pos, labels=nx.get_node_attributes(G, 'label'), with_labels=True, node_size=600, node_color="lightblue", arrows=True)
    plt.show()