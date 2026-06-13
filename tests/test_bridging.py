import numpy as np
import pytest

from bridging import (
    _best_kmeans,
    _bridging_score,
    _gradient_step,
    _project_users,
    rank_by_bridging,
    rank_by_matrix_factorization,
)


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


def test_project_users_shape():
    vm = np.array([[1, -1, 0], [0, 1, 1], [-1, 0, 1], [1, 1, -1]], dtype=float)
    coords = _project_users(vm)
    assert coords.shape == (4, 2)


def test_project_users_handles_all_zero_column():
    vm = np.array([[1, 0], [-1, 0], [1, 0]], dtype=float)
    coords = _project_users(vm)
    assert coords.shape == (3, 2)
    assert np.all(np.isfinite(coords))


def test_best_kmeans_returns_valid_labels():
    rng = np.random.default_rng(0)
    coords = np.vstack([rng.normal([-2, 0], 0.1, (20, 2)), rng.normal([2, 0], 0.1, (20, 2))])
    labels = _best_kmeans(coords, max_clusters=5)
    assert labels.shape == (40,)
    assert set(labels).issubset({0, 1, 2, 3, 4})


def test_best_kmeans_small_input_returns_single_cluster():
    coords = np.array([[1.0, 0.0], [2.0, 1.0], [0.5, -1.0]])
    labels = _best_kmeans(coords, max_clusters=5)
    assert np.all(labels == 0)


def test_bridging_score_clear_case():
    # 4 users, 2 items
    # cluster 0: users 0-1, cluster 1: users 2-3
    vm = np.array([
        [1, 1],   # user 0: likes both
        [1, 1],   # user 1: likes both
        [1, -1],  # user 2: likes A, dislikes B
        [1, -1],  # user 3: likes A, dislikes B
    ], dtype=float)
    labels = np.array([0, 0, 1, 1])
    scores = _bridging_score(vm, labels)
    assert scores[0] == 2  # A: both clusters net-positive
    assert scores[1] == 1  # B: only cluster 0 net-positive


def test_gradient_step_reduces_error():
    u_pol = np.array([0.1])
    i_pol = np.array([0.1])
    vote = 1.0
    u_int, i_int = 0.0, 0.0
    pred_before = u_int + i_int + np.dot(u_pol, i_pol)
    err_before = abs(vote - pred_before)

    new_u_int, new_i_int, new_u_pol, new_i_pol = _gradient_step(
        vote, u_int, i_int, u_pol, i_pol, lr=0.1, reg=0.0
    )
    pred_after = new_u_int + new_i_int + np.dot(new_u_pol, new_i_pol)
    err_after = abs(vote - pred_after)

    assert err_after < err_before


def test_gradient_step_returns_finite_values():
    u_pol = np.array([0.5, -0.3])
    i_pol = np.array([0.2, 0.8])
    result = _gradient_step(1.0, 0.1, -0.2, u_pol, i_pol, lr=0.01, reg=0.1)
    for val in result:
        assert np.all(np.isfinite(val))


# ---------------------------------------------------------------------------
# Integration tests — rank_by_bridging
# ---------------------------------------------------------------------------


def test_rank_by_bridging_returns_all_items():
    vm = np.array([[1, -1], [-1, 1], [1, 1]], dtype=float)
    result = rank_by_bridging(vm, ["a", "b"])
    assert len(result) == 2


def test_rank_by_bridging_sorted_descending():
    rng = np.random.default_rng(7)
    vm = rng.choice([-1, 0, 1], size=(20, 5)).astype(float)
    result = rank_by_bridging(vm, list("abcde"))
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)


def test_rank_by_bridging_bridging_item_ranked_first():
    """
    Item A: liked by both clusters. Item B: liked by only one cluster.
    Cluster 0 = users 0-2, cluster 1 = users 3-5 (clearly separated in PCA).
    """
    vm = np.array([
        # votes for [A, B]
        [1,  1],   # user 0 — cluster 0 side
        [1,  1],   # user 1
        [1,  1],   # user 2
        [1, -1],   # user 3 — cluster 1 side
        [1, -1],   # user 4
        [1, -1],   # user 5
    ], dtype=float)
    result = rank_by_bridging(vm, ["A", "B"])
    assert result[0][0] == "A"
    assert result[0][1] == 2
    assert result[1][1] == 1


def test_rank_by_bridging_item_ids_preserved():
    vm = np.array([[1, 0, -1], [-1, 1, 0]], dtype=float)
    ids = ["x", "y", "z"]
    result = rank_by_bridging(vm, ids)
    assert {iid for iid, _ in result} == set(ids)


# ---------------------------------------------------------------------------
# Integration tests — rank_by_matrix_factorization
# ---------------------------------------------------------------------------


def test_rank_by_matrix_factorization_returns_all_items():
    vm = np.array([[1, -1], [-1, 1], [1, 1]], dtype=float)
    result = rank_by_matrix_factorization(vm, ["a", "b"])
    assert len(result) == 2


def test_rank_by_matrix_factorization_sorted_descending():
    rng = np.random.default_rng(7)
    vm = rng.choice([-1, 0, 1], size=(20, 5)).astype(float)
    result = rank_by_matrix_factorization(vm, list("abcde"))
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)


def test_rank_by_matrix_factorization_consensus_item_ranked_first():
    """
    Item A: all users like it (+1). Item B: half like, half dislike (divisive).
    After training, A's item_intercept should exceed B's.
    """
    vm = np.array([
        [1,  1],
        [1,  1],
        [1,  1],
        [1,  1],
        [1, -1],
        [1, -1],
        [1, -1],
        [1, -1],
    ], dtype=float)
    result = rank_by_matrix_factorization(vm, ["A", "B"], n_epochs=500)
    assert result[0][0] == "A"


def test_rank_by_matrix_factorization_item_ids_preserved():
    vm = np.array([[1, 0, -1], [-1, 1, 0]], dtype=float)
    ids = ["x", "y", "z"]
    result = rank_by_matrix_factorization(vm, ids)
    assert {iid for iid, _ in result} == set(ids)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_item():
    vm = np.array([[1], [-1], [1]], dtype=float)
    result = rank_by_bridging(vm, ["only"])
    assert len(result) == 1
    assert result[0][0] == "only"


def test_all_abstain_item():
    vm = np.array([[0, 1], [0, -1], [0, 1]], dtype=float)
    result = rank_by_bridging(vm, ["abstain", "voted"])
    scores = dict(result)
    assert scores["abstain"] == 0


def test_single_item_matrix_factorization():
    vm = np.array([[1], [-1], [1]], dtype=float)
    result = rank_by_matrix_factorization(vm, ["only"])
    assert len(result) == 1
