import jax
import jax.numpy as jnp

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

    edge_indices = dict()
    edge_idx = 0

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

            if (i, j) not in edge_indices:
                edge_indices[(i,j)] = edge_idx
                edge_idx += 1
            if (j, i) not in edge_indices:
                edge_indices[(j,i)] = edge_idx
                edge_idx += 1

            if not alter_is_discovered:
                to_visit.append(alter_state)

    return adj_matrix, node_indices, edge_indices


def gen_random_graph(key, n_nodes=32, min_degree=2, max_degree=5):
    action_indices = {}
    adj_list = [None for i in range(n_nodes)]

    key, key1, key2 = jax.random.split(key, 3)
    degrees = jax.random.randint(key1, (n_nodes,),  min_degree, max_degree + 1)
    degree_sum = degrees.sum()
    if degrees.sum() % 2 != 0:
        adjust_idx = jax.random.randint(key2, (), 0, n_nodes)
        if degrees[adjust_idx] == min_degree:
            degrees = degrees.at[adjust_idx].set(degrees[adjust_idx] + 1)
        elif degrees[adjust_idx] == max_degree:
            degrees = degrees.at[adjust_idx].set(degrees[adjust_idx] - 1)
        else:
            adj = jax.random.choice(key2, jnp.array([-1, 1]))
            degrees = degrees.at[adjust_idx].set(degrees[adjust_idx] + adj)

    rem_degrees = jnp.copy(degrees)

    adj_matrix = jnp.zeros((n_nodes, n_nodes), dtype=jnp.uint8)
    edges = []
    edge_indices = {}
    edge_idx = 0
    while rem_degrees.sum() != 0:
        i = jnp.argmax(rem_degrees).item()
        rem_degrees_gz = rem_degrees > 0
        rem_degrees_not_argmax = jnp.arange(rem_degrees.size) != i
        pending_edges = jnp.where(rem_degrees_gz & rem_degrees_not_argmax)[0]
        key, key1 = jax.random.split(key, 2)
        j = jax.random.choice(key1, pending_edges, ()).item()

        # find a better way to do this?
        if (i, j) in edges:
            continue

        edges.append((i,j))
        edge_indices[(i,j)] = edge_idx
        edge_idx += 1
        edges.append((j,i))
        edge_indices[(j,i)] = edge_idx
        edge_idx += 1
        rem_degrees = rem_degrees.at[i].add(-1)
        rem_degrees = rem_degrees.at[j].add(-1)

    for i, j in edges:
        adj_matrix = adj_matrix.at[i, j].set(1)

    # print((jnp.sum(adj_matrix, axis=0) == degrees).all())
    # print((adj_matrix == adj_matrix.T).all())
    # print((jnp.diagonal(adj_matrix) == 0).all())

    return adj_matrix, edge_indices
