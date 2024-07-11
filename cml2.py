from operator import itemgetter

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx

from util import build_toh_adj_matrix

def make_edge_node_matrix(edge_indices, n_nodes):
    """
    edge_indices - dict{edge -> edge index}. edge is a pair (u, v) of node indices
    """
    # TODO: redo this to make it not slow
    G = jnp.zeros((len(edge_indices), n_nodes))
    for (u, v), edge_index in edge_indices.items():
        G = G.at[edge_index, u].set(1)
    return G

def recovery(x, D, theta):
    # D ~ h x n, dictionary of column vectors. assume hyperdimensional, so all the columns are pseudo-orthogonal
    # x ~ h x 1, column vectors
    # theta: positive scalar
    # returns: d_i if there d_i is a row vector such that its cosine similarity with x is > theta, else 0
    x_unit = x / jnp.linalg.norm(x)
    D_sim = (D / jnp.linalg.norm(D, axis=0)).T @ x_unit

    if jnp.any(D_sim >= theta):
        col = jnp.argmax(D_sim)
        return D[:, col], col.item()
    else:
        return jnp.zeros_like(x), None

def winner_take_all(x):
    print(x)
    win_idx = jnp.argmax(x)
    print(f"{win_idx=}")
    return jnp.zeros_like(x).at[win_idx].set(x[win_idx])


def calc_noise_floor(key, emb_dim, num_trials):
    key, *subkeys1 = jax.random.split(key, num=num_trials + 1)
    subkeys2 = jax.random.split(key, num=num_trials)
    subkeys = zip(subkeys1, subkeys2)
    max_sim = 0.0
    max_sim_bip = 0.0
    sims = []
    for key1, key2 in subkeys:
        x = jax.random.normal(key1, (emb_dim,))
        y = jax.random.normal(key2, (emb_dim,))
        sim = jnp.dot(x, y) / (jnp.linalg.norm(x) * jnp.linalg.norm(y))
        sim = sim.item()
        sims.append(sim)

        if sim > max_sim:
            max_sim = sim

        x_bip = jnp.sign(x)
        y_bip = jnp.sign(y)
        sim_bip = jnp.dot(x_bip, y_bip) / (jnp.linalg.norm(x_bip) * jnp.linalg.norm(y_bip))
        sim_bip = sim_bip.item()

        if sim_bip > max_sim_bip:
            max_sim_bip = sim_bip

    print(f"{max_sim=}, {max_sim_bip=}")

    return max_sim, sims


class McDonaldCML:
    def __init__(self, key, params, G, edge_lookup, node_lookup):
        param_names = ['n_obs', 'n_act', 'emb_dim', 'S_init_stddev', 'A_init_stddev', 'lr_s', 'lr_a']
        n_obs, n_act, emb_dim, S_init_stddev, A_init_stddev, lr_s, lr_a = itemgetter(*param_names)(params)

        self.emb_dim = emb_dim
        self.n_obs = n_obs
        self.n_act = n_act
        self.lr_s = lr_s
        self.lr_a = lr_a

        key, key1, key2 = jax.random.split(key, 3)

        self.S = jax.random.normal(key1, (emb_dim, n_obs)) * S_init_stddev
        self.A = jax.random.normal(key2, (emb_dim, n_act)) * A_init_stddev
        self.G = G
        self.edge_lookup = edge_lookup
        self.node_lookup = node_lookup
        self.A_pinv = None


    def step(self, s_curr, s_goal, s_goal_thresh, term_thresh):
        # If the target state s* was not
        # in S, then it would be pseudo-orthogonal to all node states, δ(s*,
        # S) < θ, and the CML returned a zeros vector (the MAP identity
        # element under addition); else if δ(s*, S) ≥ θ, then the raw s*
        # vector was used.
        s_goal_sims = self.S.T @ s_goal
        print(f"{s_goal_sims=}")
        s_goal_matches = (self.S.T @ s_goal >= s_goal_thresh).any()

        if not s_goal_matches:
            print("Goal mismatch")
            print(s_goal)
            return jnp.zeros_like(s_goal)

        # A cleanup operation, (13), was always applied to the current node state
        # to mitigate state drift errors
        s_curr_rec, s_idx = recovery(s_curr, self.S, s_goal_thresh)
        print(f"{s_idx=}")

        # When the target and current node state were sufficiently similar,
        # δ(s*, s_t) ≥ ϕ, then the CML returned s_t.
        sim = jnp.dot(s_goal, s_curr_rec) / (jnp.linalg.norm(s_goal) * jnp.linalg.norm(s_curr_rec))
        print(f"{sim=}")
        if sim >= term_thresh:
            return s_curr_rec

        s_diff = s_goal - s_curr_rec

        u = self.A_pinv @ s_diff
        g = self.G[:, s_idx]

        g_nz_idxs = jnp.where(g == 1)[0].tolist()
        for e_idx in g_nz_idxs:
            n1, n2 = self.edge_lookup[e_idx]
            print(f"{(n1, n2)=} {self.node_lookup[n1]=}, {self.node_lookup[n2]=}, {u[e_idx]=}")
        print(f"g nonzero edges = {[self.edge_lookup[e_idx] for e_idx in g_nz_idxs]}")

        print(u[g_nz_idxs])
        print(g)
        print(u.mean(), u.min(), u.max())
        ug = u * g

        print(ug.mean(), ug.min(), ug.max())
        c = winner_take_all(u * g)

        print(c != 0)

        s_pred = s_curr_rec + self.A @ c

        return s_pred

        # WTA

    def learn(self, num_epochs, normalize=False, learn_A=True):
        # in the nature paper, they generate 200 random walks and keep iterate for 10 epochs
        # here, the paper just says "One training epoch spans all e actions"
        # if iterating through edges, this fixes both the pre- and post-edge node
        # there's a single learning rate, α = 0.1
        # also, "For simplicity, weight updates are summed and applied at the end of each training epoch.""
        mspes = []
        s_col_norms = []
        s_norms = []
        for epoch in range(num_epochs):
            # ŝ_{t+1} = s_t + A c_t
            # ΔS(t) = α (ŝ_{t+1} – s{t+1}) o_{t+1}^T
            # ΔA(t) = α (s_{t+1} – ŝ_{t+1}) c_t^T

            edges = jnp.arange(self.n_act)
            nodes_pre, nodes_post = zip(*(self.edge_lookup[e] for e in range(self.n_act)))
            nodes_pre = jnp.array(nodes_pre)
            nodes_post = jnp.array(nodes_post)

            s_hat = self.S[:, nodes_pre] + self.A[:, edges]
            s_post = self.S[:, nodes_post]
            pred_errs = s_hat - s_post
            self.S = self.S.at[:, nodes_post].add(self.lr_s * pred_errs)
            self.A = self.A.at[:, edges].add(-self.lr_s * pred_errs)
            mspes.append((pred_errs ** 2).mean())

            S_norm = jnp.linalg.norm(self.S)
            s_col_norms.append(jnp.linalg.norm(self.S, axis=0))
            s_norms.append(S_norm)

            if normalize:
                # normalize (equation (5))
                # self.S = self.S.at[:].divide(S_norm**2)
                self.S = self.S.at[:].divide(jnp.linalg.norm(self.S, axis=0))

                # not clear why paper specifies this, especially with discussion of implicit
                # regularization in the Nature paper. TODO: test
                self.A = self.A.at[:].divide(jnp.linalg.norm(self.A, axis=0))

            # delta_S = jnp.zeros_like(self.S)
            # delta_A = jnp.zeros_like(self.A)
            # for act_idx in range(self.n_act):
            #     a = self.A[:, act_idx]
            #     o_pre, o_post = self.edge_lookup[act_idx]
            #     s_hat = self.S[:, o_pre] + a
            #     s_post = self.S[:, o_post]
            #     pred_err = s_hat - s_post
                # delta_S = delta_S.at[:, o_post].add(self.lr_s * pred_err)
                # delta_A = delta_A.at[:, act_idx].add(-self.lr_a * pred_err)
                # self.S = self.S.at[:, o_post].add(self.lr_s * pred_err)
                # self.A = self.A.at[:, act_idx].add(-self.lr_a * pred_err)
                # mspes.append((pred_err ** 2).mean())

        # s_col_norms = jnp.array(s_col_norms)

        # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
        # eps = range(num_epochs)
        # for i in range(self.n_obs):
        #     ax[0].plot(eps[:2], s_col_norms[:2, i], label=f'col {i}')
        # ax[0].legend()
        # ax[1].plot(eps, s_norms)
        # plt.show()
        # plt.close(fig)

        
        self.A_pinv = jnp.linalg.pinv(self.A)

        # calculated (not learned) A
        if not learn_A:
            self.A_calc = jnp.empty((self.emb_dim, self.n_act))
            mses_A_learn_calc = []
            for a in range(self.n_act):
                o_pre, o_post = self.edge_lookup[a]
                col = self.S[:, o_pre] - self.S[:, o_post]
                self.A_calc = self.A_calc.at[:, a].set(col)
                mses_A_learn_calc.append(((col - self.A[:, a])**2).mean())

            print(f"{sum(mses_A_learn_calc)/len(mses_A_learn_calc)}")
            self.A = self.A_calc

        return mspes



if __name__ == '__main__':
    seed = 1234
    key = jax.random.PRNGKey(seed)

    emb_dim = 1000

    # key, subkey = jax.random.split(key, 2)
    # theta, sims = calc_noise_floor(subkey, emb_dim, num_trials=20000)
    # fig, ax = plt.subplots()
    # ax.hist(sims, bins=30, density=True)
    # ax.set_title(f'Distribution of similarities for random pairs, emb_dim = {emb_dim}')
    # plt.show()

    adj_matrix, node_indices, edge_indices = build_toh_adj_matrix()
    node_lookup = { node_idx : node for node, node_idx in node_indices.items() }
    edge_lookup = { edge_idx : edge for edge, edge_idx in edge_indices.items() }
    n_nodes = adj_matrix.shape[0]

    # adj_matrix_np = jnp.asarray(adj_matrix)
    # G = nx.from_numpy_array(adj_matrix_np, create_using=nx.DiGraph)
    # labels = { node_idx : f'{s[0]}{s[1]}{s[2]}' for s, node_idx in node_indices.items() }
    # nx.set_node_attributes(G, labels, 'label')
    # pos = nx.spring_layout(G)  # positions for all nodes
    # nx.draw(G, pos, labels=nx.get_node_attributes(G, 'label'), with_labels=True, node_size=600, node_color="lightblue", arrows=True)
    # plt.show()

    G = make_edge_node_matrix(edge_indices, n_nodes)

    cml_params = {
        'n_obs': n_nodes,
        'n_act': len(edge_indices),
        'emb_dim': emb_dim,
        'S_init_stddev': 1.0,
        'A_init_stddev': 0.1,
        'lr_s': 0.1,
        'lr_a': 0.01 # TODO: McDonald paper uses 0.1 for both
    }

    key, subkey = jax.random.split(key, 2)
    cml = McDonaldCML(subkey, cml_params, G, edge_lookup, node_lookup)
    num_epochs = 30
    pred_errs = cml.learn(num_epochs)

    # fig, ax = plt.subplots()
    # ax.plot(pred_errs)
    # ax.set_title('mean squared prediction errors during training')
    # ax.set_yscale('log')
    # plt.show()
    # plt.close(fig)

    # calculated from noise floor
    s_goal_thresh = 0.1

    # the paper says this is "experimentally determined", but doesn't comment
    # on how exactly
    term_thresh = 0.3

    key, subkey = jax.random.split(key, 2)
    o_init, o_goal = jax.random.choice(subkey, jnp.arange(n_nodes), (2,), replace=False).tolist()
    o_init = 0
    print(f"{o_init=} [{node_lookup[o_init]}], {o_goal=} [{node_lookup[o_goal]}]")
    s_curr = cml.S[:, o_init]
    s_goal = cml.S[:, o_goal]

    print(cml.S.T @ cml.S[:, 0])


    num_steps = 0
    while True:
        print('--------------------------')
        s_next = cml.step(s_curr, s_goal, s_goal_thresh, term_thresh)
        s_next_rec, s_next_rec_idx = recovery(s_next, cml.S, s_goal_thresh)
        print(f"{s_next_rec_idx=}")
        num_steps += 1
        if jnp.isclose(s_next, s_goal).all():
            break
        raise NotImplementedError

    print(f"{num_steps=}")
