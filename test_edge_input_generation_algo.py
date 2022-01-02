from networkx import gnp_random_graph
from networkx.linalg.graphmatrix import adjacency_matrix
from tqdm import tqdm
from bfs_numba_impl import bfs_numba_spatial_pos_and_edge_input
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
        p = np.random.uniform(0.25, 0.75)
        graph = gnp_random_graph(n=n, p=p)
        adj_matrix = adjacency_matrix(graph).toarray()
        adj_matrixs.append(adj_matrix.astype("long"))

    return adj_matrixs

def generate_edge_feats(adj_matrixs, seed=42):

    np.random.seed(seed)

    print("[Data] Edge Feature Preparing ...")

    edge_feats = []
    for adj in tqdm(adj_matrixs):
        edge_shp = (adj.shape[0], adj.shape[1], 3)
        edge_feat = np.random.randint(0, 5, edge_shp)
        edge_feat = edge_feat.astype("int64")
        edge_feat *= np.expand_dims(adj, -1)
        edge_feats.append(edge_feat.astype("long"))

    return edge_feats

def compare(a, b):
    return (a==b).all()

def test_algo_correctness(adj_matrixs, edge_feats, max_dist=5):

    N = len(adj_matrixs)
    assert len(edge_feats) == N

    for i in tqdm(range(N)):

        adj, edge = adj_matrixs[i], edge_feats[i]

        # fw algo
        res_fw = algos.fw_spatial_pos_and_edge_input(adj, edge, max_dist)

        # bfs algo
        res_cython = algos.bfs_spatial_pos_and_edge_input(adj, edge, max_dist)

        # bfs (numba) algo
        res_nb = bfs_numba_spatial_pos_and_edge_input(adj, edge, max_dist)

        assert compare(res_fw[0], res_cython[0])
        # since shortest path is not unique
        # thus the following check is ommitted:
        # assert compare(edge_input_fw, res[1])

        assert compare(res_cython[0], res_nb[0])
        assert compare(res_cython[1], res_nb[1])

def benchmark_old(adj_matrixs, edge_feats):
    cur = datetime.now()

    N = len(adj_matrixs)
    for i in range(N):
        adj, edge = adj_matrixs[i], edge_feats[i]
        res_fw = algos.fw_spatial_pos_and_edge_input(adj, edge, 5)

    print("[Benchmark] Time Count for Floyd: ", datetime.now() - cur)

def benchmark_new(adj_matrixs, edge_feats):
    cur = datetime.now()

    N = len(adj_matrixs)
    for i in range(N):
        adj, edge = adj_matrixs[i], edge_feats[i]
        res = algos.bfs_spatial_pos_and_edge_input(adj, edge, 5)

    print("[Benchmark] Time Count for BFS: ", datetime.now() - cur)

def benchmark_numba(adj_matrixs, edge_feats):
    cur = datetime.now()

    N = len(adj_matrixs)
    for i in range(N):
        adj, edge = adj_matrixs[i], edge_feats[i]
        res = bfs_numba_spatial_pos_and_edge_input(adj, edge, 5)

    print("[Benchmark] Time Count for BFS (Numba): ", datetime.now() - cur)

def main():

    adj_matrixs = generate_adj_matrixs(N=3000)
    edge_feats = generate_edge_feats(adj_matrixs)

    print("[Testing] Test Correctness ...")

    test_algo_correctness(adj_matrixs, edge_feats)

    print("[Benchmark] Start ...")

    benchmark_old(adj_matrixs, edge_feats)
    benchmark_new(adj_matrixs, edge_feats)
    benchmark_numba(adj_matrixs, edge_feats)

if __name__ == "__main__":

    main()
