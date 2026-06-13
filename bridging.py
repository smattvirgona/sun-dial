"""
Bridging-based content ranking algorithms.

Two approaches drawn from Audrey Tang / Plurality research:
  - rank_by_bridging: Polis-style (PCA + k-means + cluster consensus)
  - rank_by_matrix_factorization: Community Notes-style (polarity-aware MF)

Vote matrix convention: shape (n_users, n_items), values in {-1, 0, 1}.
Zero means no vote / pass.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def rank_by_bridging(
    vote_matrix: np.ndarray,
    item_ids: list,
    max_clusters: int = 5,
) -> list:
    """
    Rank items by how many distinct opinion clusters gave them net-positive votes.

    Parameters
    ----------
    vote_matrix : np.ndarray, shape (n_users, n_items)
        Values in {-1, 0, 1}. 0 = no vote.
    item_ids : list, length n_items
    max_clusters : int
        Upper bound on k for k-means; best k chosen by silhouette score.

    Returns
    -------
    list of (item_id, bridging_score) sorted descending by bridging_score,
    then descending by total votes cast as a tiebreaker.
    """
    vote_matrix = np.asarray(vote_matrix, dtype=float)
    coords = _project_users(vote_matrix)
    labels = _best_kmeans(coords, max_clusters)
    scores = _bridging_score(vote_matrix, labels)
    total_votes = np.abs(vote_matrix).sum(axis=0)
    ranking = sorted(
        zip(item_ids, scores, total_votes),
        key=lambda x: (x[1], x[2]),
        reverse=True,
    )
    return [(iid, int(s)) for iid, s, _ in ranking]


def rank_by_matrix_factorization(
    vote_matrix: np.ndarray,
    item_ids: list,
    n_factors: int = 1,
    n_epochs: int = 200,
    lr: float = 0.01,
    reg: float = 0.1,
) -> list:
    """
    Rank items by cross-partisan appeal using a Community Notes-style model.

    Model: vote ≈ user_intercept + item_intercept + dot(user_polarity, item_polarity)

    item_intercept captures helpfulness independent of the user's polarity.
    Items with high item_intercept are liked across ideological divides.

    Parameters
    ----------
    vote_matrix : np.ndarray, shape (n_users, n_items)
        Values in {-1, 0, 1}. 0 = no vote.
    item_ids : list, length n_items
    n_factors : int
        Polarity vector dimensionality. 1 = single left-right axis (default).
    n_epochs : int
        Training passes over observed votes.
    lr : float
        SGD learning rate.
    reg : float
        L2 regularisation coefficient.

    Returns
    -------
    list of (item_id, item_intercept) sorted descending by item_intercept.
    """
    vote_matrix = np.asarray(vote_matrix, dtype=float)
    n_users, n_items = vote_matrix.shape

    np.random.seed(42)
    u_int = np.random.randn(n_users) * 0.01
    i_int = np.random.randn(n_items) * 0.01
    u_pol = np.random.randn(n_users, n_factors) * 0.01
    i_pol = np.random.randn(n_items, n_factors) * 0.01

    user_idxs, item_idxs = np.nonzero(vote_matrix)

    for _ in range(n_epochs):
        order = np.random.permutation(len(user_idxs))
        for k in order:
            u, i = user_idxs[k], item_idxs[k]
            u_int[u], i_int[i], u_pol[u], i_pol[i] = _gradient_step(
                vote_matrix[u, i], u_int[u], i_int[i], u_pol[u], i_pol[i], lr, reg
            )

    ranking = sorted(zip(item_ids, i_int), key=lambda x: x[1], reverse=True)
    return [(iid, float(score)) for iid, score in ranking]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _project_users(vote_matrix: np.ndarray) -> np.ndarray:
    """
    Project users into 2D opinion space via PCA.

    Missing votes (0) are imputed with per-item column means of observed votes
    before PCA so absent votes don't pull users toward the "disagree" pole.
    """
    imputed = vote_matrix.copy()
    for j in range(vote_matrix.shape[1]):
        col = vote_matrix[:, j]
        observed = col[col != 0]
        mean = observed.mean() if len(observed) > 0 else 0.0
        imputed[col == 0, j] = mean

    n_components = min(2, imputed.shape[0], imputed.shape[1])
    coords = PCA(n_components=n_components).fit_transform(imputed)

    # Pad to 2 columns if the matrix was too small for 2 components
    if coords.shape[1] < 2:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 2 - coords.shape[1]))])
    return coords


def _best_kmeans(user_coords: np.ndarray, max_clusters: int) -> np.ndarray:
    """
    Run k-means for k in [2, max_clusters]; return labels for the best-scoring k.

    Falls back to a single cluster (all zeros) when there are fewer than 4 users,
    since silhouette scoring requires at least 2 samples per cluster.
    """
    n = user_coords.shape[0]
    if n < 4:
        return np.zeros(n, dtype=int)

    best_labels = np.zeros(n, dtype=int)
    best_score = -1.0
    k_max = min(max_clusters, n - 1)

    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(user_coords)
        score = silhouette_score(user_coords, labels)
        if score > best_score:
            best_score = score
            best_labels = labels

    return best_labels


def _bridging_score(
    vote_matrix: np.ndarray, cluster_labels: np.ndarray
) -> np.ndarray:
    """
    For each item, count the number of clusters where net votes > 0.
    """
    n_items = vote_matrix.shape[1]
    clusters = np.unique(cluster_labels)
    scores = np.zeros(n_items, dtype=int)

    for c in clusters:
        mask = cluster_labels == c
        net = vote_matrix[mask].sum(axis=0)
        scores += (net > 0).astype(int)

    return scores


def _gradient_step(
    vote: float,
    u_int: float,
    i_int: float,
    u_pol: np.ndarray,
    i_pol: np.ndarray,
    lr: float,
    reg: float,
) -> tuple:
    """
    Apply one SGD update for a single observed vote.

    Returns updated (u_int, i_int, u_pol, i_pol).
    """
    pred = u_int + i_int + np.dot(u_pol, i_pol)
    err = vote - pred

    new_u_int = u_int + lr * (err - reg * u_int)
    new_i_int = i_int + lr * (err - reg * i_int)
    new_u_pol = u_pol + lr * (err * i_pol - reg * u_pol)
    new_i_pol = i_pol + lr * (err * u_pol - reg * i_pol)

    return new_u_int, new_i_int, new_u_pol, new_i_pol
