from operator import itemgetter

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx

class CML:
    def __init__(self, key, params):
        param_names = ['n_obs', 'n_act', 'emb_dim', 'Q_init_stddev', 'V_init_stddev', 'W_init_stddev', 'eta_q', 'eta_v', 'eta_w']
        n_obs, n_act, emb_dim, Q_init_stddev, V_init_stddev, W_init_stddev, eta_q, eta_v, eta_w = itemgetter(*param_names)(params)

        self.eta_q = eta_q
        self.eta_v = eta_v
        self.eta_w = eta_w

        key, key1, key2, key3 = jax.random.split(key, 4)
        self.Q = jax.random.normal(key1, (emb_dim, n_obs)) * Q_init_stddev
        self.V = jax.random.normal(key2, (emb_dim, n_act)) * V_init_stddev
        self.W = jax.random.normal(key3, (n_act, emb_dim)) * W_init_stddev

        self.I_A = jnp.eye(n_act)

    def learn_from_trajectories(self, trajectories, num_epochs):
        # O : dimension of observation space
        # A : dimension of action space
        # D : dimension of embedding space
        # L : length of each trajectory
        num_trajectories = trajectories.shape[0]
        mses = []
        for epoch in range(num_epochs):
            print(f"epoch {epoch}")
            for traj_idx in range(num_trajectories):
                nodes = trajectories[traj_idx, :, 0]
                edges = trajectories[traj_idx, :, 1]
                next_nodes = trajectories[traj_idx, :, 2]

                a_AxL = self.I_A[:, edges]

                s_curr_DxL = self.Q[:, nodes]
                s_next_DxL = self.Q[:, next_nodes]

                s_diff_DxL = s_next_DxL - s_curr_DxL
                pred_err_DxL = s_diff_DxL - self.V[:, edges]

                self.V = self.V.at[:, edges].add(self.eta_v * pred_err_DxL)
                self.Q = self.Q.at[:, next_nodes].add(- self.eta_q * pred_err_DxL)
                self.W = self.W.at[edges, :].add(self.eta_w * s_diff_DxL.T)
                mses.append((pred_err_DxL ** 2).mean())
        return mses



    def action_similarities(self):
        return self.V.T @ self.V




    
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

def draw_graph(adj_list):
    G = nx.Graph()
    for i, neighbors in enumerate(adj_list):
        for neighbor in neighbors:
            G.add_edge(i, neighbor.item())
    
    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.show()

def do_graph_random_walks(adj_matrix, edge_indices, num_walks, walk_length, key):
    num_nodes = adj_matrix.shape[0]
    nodes = jnp.arange(num_nodes)
    trajectories = []
    for i in range(num_walks):
        key, key1 = jax.random.split(key)
        trajectory = []
        curr_node = jax.random.choice(key1, nodes).item()
        for j in range(walk_length):
            key, key1 = jax.random.split(key)
            next_node = jax.random.choice(key, jnp.where(adj_matrix[curr_node] == 1)[0]).item()
            trajectory.append((curr_node, edge_indices[(curr_node, next_node)], next_node))
            curr_node = next_node
        trajectories.append(trajectory)
    return jnp.array(trajectories)



if __name__ == '__main__':
    seed = 1234
    key = jax.random.PRNGKey(seed)

    adj_matrix, edge_indices = gen_random_graph(key)
    # draw_graph(adj_list)

    # take 200 random walks of length 32, each initialized from a random starting point
    # save them to do replay
    num_walks = 200
    walk_length = 32
    trajectories = do_graph_random_walks(adj_matrix, edge_indices, num_walks, walk_length, key)

    n_obs = adj_matrix.shape[0]
    n_act = len(edge_indices)
    emb_dim = 1000
    Q_init_stddev = 1.0
    V_init_stddev = 0.1
    W_init_stddev = 0.1
    eta_q = 0.1
    eta_v = 0.01
    eta_w = 0.01
    cml_params = {
        'n_obs': n_obs,
        'n_act': n_act,
        'emb_dim': emb_dim,
        'Q_init_stddev': Q_init_stddev,
        'V_init_stddev': V_init_stddev,
        'W_init_stddev': W_init_stddev,
        'eta_q': eta_q,
        'eta_v': eta_v,
        'eta_w': eta_w,
    }
    cml = CML(key, cml_params)
    num_train_epochs = 10
    pred_errors = cml.learn_from_trajectories(trajectories, num_train_epochs)

    fig, ax = plt.subplots()
    ax.plot(pred_errors)
    ax.set_title('Prediction errors during training')
    ax.set_yscale('log')
    plt.show()
    plt.close(fig)

    cos_sims = cml.action_similarities()

    fig, ax = plt.subplots()
    im = ax.imshow(cos_sims, cmap='plasma')
    ax.set_title("Cosine similarities, V matrix")
    fig.colorbar(im)
    fig.tight_layout()
    plt.show()
    plt.close(fig)
