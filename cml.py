from operator import itemgetter

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx

from util import build_toh_adj_matrix, gen_random_graph

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
        V_norm = self.V / jnp.linalg.norm(self.V, axis=1, keepdims=True)
        return V_norm.T @ V_norm




    
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
    return jnp.array(trajectories), key







if __name__ == '__main__':
    seed = 1234
    key = jax.random.PRNGKey(seed)

    graph_type = 'ToH'

    if graph_type == 'rand':
        adj_matrix, edge_indices = gen_random_graph(key)

        # take 200 random walks of length 32, each initialized from a random starting point
        # save them to do replay
        num_walks = 200
        walk_length = 32
        trajectories, key = do_graph_random_walks(adj_matrix, edge_indices, num_walks, walk_length, key)
    elif graph_type == 'ToH':
        adj_matrix, node_indices, edge_indices = build_toh_adj_matrix()
    else:
        raise Exception(f"unrecognized graph_type value '{graph_type}'")

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

    def cml_matrices_are_pseudo_orthogonal():
        cos_sims = cml.action_similarities()
        fig, ax = plt.subplots()
        im = ax.imshow(cos_sims, cmap='plasma')
        ax.set_title("Cosine similarities, V matrix")
        fig.colorbar(im)
        fig.tight_layout()
        plt.show()
        plt.close(fig)

        # distance between pairs of state nodes versus the shortest path length between said states in the graph
        G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
        print(nx.is_strongly_connected(G))
        print(nx.is_weakly_connected(G))
        state_distances = {}
        for i in range(n_obs):
            for j in range(i + 1, n_obs):
                shpath = nx.shortest_path(G, source=i, target=j)
                shpath_len = len(shpath)
                state_node_dist = jnp.linalg.norm(cml.Q[:, i] - cml.Q[:, j])
                if shpath_len not in state_distances:
                    state_distances[shpath_len] = []
                state_distances[shpath_len].append(state_node_dist)

        for spl in state_distances:
            state_distances[spl] = sum(state_distances[spl]) / len(state_distances[spl])

        fig, ax = plt.subplots()
        ax.bar(state_distances.keys(), state_distances.values())
        plt.show()
        plt.close(fig)


    # the "Assembling Modular, Hierarchical Cognitive Map Learners with Hyperdimensional Computing"
    # paper says "vector element values [are] normally distributed over [-0.1, 0.1]"
    # check this by scatter-plotting vector elements
    def state_vectors_normally_distrib():
        Q_flat = cml.Q.reshape(-1)
        fig, ax = plt.subplots()
        ax.hist(Q_flat, bins=100, density=True)

        def plot_gaussian(ax, gauss_mean, gauss_std, plot_color):
            plot_xs = jnp.linspace(gauss_mean - 3 * gauss_std, gauss_mean + 3 * gauss_std, 200)
            plot_ys = 1.0 / (jnp.sqrt(2 * jnp.pi) * gauss_std) * jnp.exp(-0.5 * ((plot_xs - gauss_mean) / gauss_std)**2)
            ax.plot(plot_xs, plot_ys, color=plot_color)

        plot_gaussian(ax, 0.0, 0.35, 'r')
        plt.show()
        plt.close(fig)

    # cml_matrices_are_pseudo_orthogonal()
    # state_vectors_normally_distrib()
