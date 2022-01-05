from networkx import gnp_random_graph
from networkx.linalg.graphmatrix import adjacency_matrix
from tqdm import tqdm
from algos_numba import (
    bfs_numba_spatial_pos_and_edge_input,
    bfs_numba_target_spatial_pos_and_edge_input
)
from datetime import datetime
import numpy as np
import algos


def generate_adj_matrixs(N=1, seed=42):

    np.random.seed(seed)

    print("[Data] Total Matrix Counts:", N)
    print("[Data] Matrix Preparing ...")

    adj_matrixs = []
    for _ in tqdm(range(N)):
        n = np.random.randint(20, 100)
        p = np.random.uniform(0.05, 0.5)
        graph = gnp_random_graph(n=n, p=p)
        adj_matrix = adjacency_matrix(graph).toarray()
        adj_matrixs.append(adj_matrix.astype("int64"))

    return adj_matrixs

def generate_edge_feats(adj_matrixs, seed=42):

    np.random.seed(seed)

    print("[Data] Edge Feature Preparing ...")

    edge_feats = []
    for adj in tqdm(adj_matrixs):
        edge_shp = (adj.shape[0], adj.shape[1], 3)
        edge_feat = np.random.randint(0, 5, edge_shp)
        edge_feat = edge_feat
        edge_feat *= np.expand_dims(adj, -1)
        edge_feats.append(edge_feat.astype("int64"))

    return edge_feats

def compare(a, b):
    return (a==b).all()

def test_source_algo_correctness(adj_matrixs, edge_feats, max_dist=5):

    N = len(adj_matrixs)
    assert len(edge_feats) == N

    for i in tqdm(range(N)):

        adj, edge = adj_matrixs[i], edge_feats[i]

        # fw algo
        (
            shorest_path, edge_input
        ) = algos.fw_spatial_pos_and_edge_input(adj, edge, max_dist)

        # bfs algo
        res = algos.get_source_spatial_pos_and_edge_input(adj, edge, max_dist)

        # bfs (numba) algo
        res_nb = bfs_numba_spatial_pos_and_edge_input(adj, edge, max_dist)

        assert compare(shorest_path, res[0])
        # since shortest path is not unique
        # thus the following check is ommitted:
        # assert compare(edge_input_fw, res[1])

        assert compare(res[0], res_nb[0])
        assert compare(res[1], res_nb[1])

def test_target_algo_correctness(adj_matrixs, edge_feats, max_dist=5):

    N = len(adj_matrixs)
    assert len(edge_feats) == N

    for i in tqdm(range(N)):

        adj, edge = adj_matrixs[i], edge_feats[i]

        # bfs algo
        res = algos.get_target_spatial_pos_and_edge_input(
            adj,
            edge,
            max_dist
        )

        # bfs (numba) algo
        res_nb = bfs_numba_target_spatial_pos_and_edge_input(
            adj,
            edge,
            max_dist
        )

        assert compare(res[0], res_nb[0])
        assert compare(res[1], res_nb[1])

def benchmark_source_floyd(adj_matrixs, edge_feats, max_dist=5):
    cur = datetime.now()

    N = len(adj_matrixs)
    for i in range(N):
        adj, edge = adj_matrixs[i], edge_feats[i]
        res = algos.fw_spatial_pos_and_edge_input(adj, edge, max_dist)

    print("[Benchmark] Time Count for Floyd: ", datetime.now() - cur)

def benchmark_source_bfs(adj_matrixs, edge_feats, max_dist=5):
    cur = datetime.now()

    N = len(adj_matrixs)
    for i in range(N):
        adj, edge = adj_matrixs[i], edge_feats[i]
        res = algos.get_target_spatial_pos_and_edge_input(adj, edge, max_dist)

    print("[Benchmark] Time Count for BFS: ", datetime.now() - cur)

def benchmark_source_bfs_numba(adj_matrixs, edge_feats, max_dist=5):
    cur = datetime.now()

    N = len(adj_matrixs)
    for i in range(N):
        adj, edge = adj_matrixs[i], edge_feats[i]
        res = bfs_numba_spatial_pos_and_edge_input(adj, edge, max_dist)

    print("[Benchmark] Time Count for BFS (Numba): ", datetime.now() - cur)

def benchmark_target_bfs(adj_matrixs, edge_feats, max_dist=5):
    cur = datetime.now()

    N = len(adj_matrixs)
    for i in range(N):
        adj, edge = adj_matrixs[i], edge_feats[i]
        res = algos.get_source_spatial_pos_and_edge_input(adj, edge, max_dist)

    print("[Benchmark] Time Count for BFS: ", datetime.now() - cur)

def benchmark_target_bfs_numba(adj_matrixs, edge_feats, max_dist=5):
    cur = datetime.now()

    N = len(adj_matrixs)
    for i in range(N):
        adj, edge = adj_matrixs[i], edge_feats[i]
        res = bfs_numba_target_spatial_pos_and_edge_input(adj, edge, max_dist)

    print("[Benchmark] Time Count for BFS (Numba): ", datetime.now() - cur)

def main():

    adj_matrixs = generate_adj_matrixs(N=3000)
    edge_feats = generate_edge_feats(adj_matrixs)

    print("[Testing] Test Source Correctness ...")

    test_source_algo_correctness(adj_matrixs, edge_feats)

    print("[Testing] Test Target Correctness ...")

    test_target_algo_correctness(adj_matrixs, edge_feats)

    print("[Source Benchmark] Start ...")

    benchmark_source_floyd(adj_matrixs, edge_feats)
    benchmark_source_bfs(adj_matrixs, edge_feats)
    benchmark_source_bfs_numba(adj_matrixs, edge_feats)

    print("[Target Benchmark] Start ...")

    benchmark_target_bfs(adj_matrixs, edge_feats)
    benchmark_target_bfs_numba(adj_matrixs, edge_feats)

if __name__ == "__main__":

    main()
