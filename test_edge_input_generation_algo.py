from networkx import gnp_random_graph
from networkx.linalg.graphmatrix import adjacency_matrix
from tqdm import tqdm
from bfs_numba_impl import numba_bfs
from datetime import datetime
import numpy as np
import algos

"""
# current cython bad case:

adj = np.array([[0,1,0,0,1],
                [1,0,1,0,0],
                [0,1,0,1,0],
                [0,0,1,0,1],
                [1,0,0,1,0]]).astype("int32")
n = adj.shape[0]
edge = np.arange(n*n*3).reshape(n,n,3)
edge = edge * np.expand_dims(arr, -1)

max_dist = 2
shortest_path, path = algos.floyd_warshall(arr)
edge_input = algos.gen_edge_input(max_dist, path, edge)

# please check edge_input[-1, 1]
# this should be:
array([[60, 61, 62],
       [ 3,  4,  5]])
# instead of:
array([[ 0,  0,  0],
       [-1, -1, -1]])

# the reason for this is we should set original value of path
# to -1 instead of 0, since 0 was choosed to act as the stop
# condition in get_all_edges, which causes a wrong path generation
"""

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
        adj_matrixs.append(adj_matrix.astype("int32"))

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
        edge_feats.append(edge_feat)

    return edge_feats

def compare(a, b):
    return (a==b).all()

def test_algo_correctness(adj_matrixs, edge_feats, max_dist=5):

    N = len(adj_matrixs)
    assert len(edge_feats) == N

    for i in tqdm(range(N)):

        adj, edge = adj_matrixs[i], edge_feats[i]

        # fw algo
        shorest_path, path = algos.floyd_warshall(adj)
        edge_input_fw = algos.gen_edge_input(max_dist, path, edge)

        # bfs algo
        res = algos.get_source_spatial_pos_and_edge_input(adj, edge, max_dist)

        # bfs (numba) algo
        res_nb = numba_bfs(adj, edge, max_dist)

        assert compare(shorest_path, res[0])
        # since shortest path is not unique
        # thus the following check is ommitted:
        # assert compare(edge_input_fw, res[1])

        assert compare(res[0], res_nb[0])
        assert compare(res[1], res_nb[1])

def benchmark_old(adj_matrixs, edge_feats, max_dist=5):
    cur = datetime.now()

    N = len(adj_matrixs)
    for i in range(N):
        adj, edge = adj_matrixs[i], edge_feats[i]
        shorest_path, path = algos.floyd_warshall(adj)
        edge_input_fw = algos.gen_edge_input(max_dist, path, edge)

    print("[Benchmark] Time Count for Floyd: ", datetime.now() - cur)

def benchmark_new(adj_matrixs, edge_feats, max_dist=5):
    cur = datetime.now()

    N = len(adj_matrixs)
    for i in range(N):
        adj, edge = adj_matrixs[i], edge_feats[i]
        res = algos.get_source_spatial_pos_and_edge_input(adj, edge, max_dist)

    print("[Benchmark] Time Count for BFS: ", datetime.now() - cur)

def benchmark_numba(adj_matrixs, edge_feats, max_dist=5):
    cur = datetime.now()

    N = len(adj_matrixs)
    for i in range(N):
        adj, edge = adj_matrixs[i], edge_feats[i]
        res = numba_bfs(adj, edge, max_dist)

    print("[Benchmark] Time Count for BFS (Numba): ", datetime.now() - cur)

def main():

    adj_matrixs = generate_adj_matrixs(N=5000)
    edge_feats = generate_edge_feats(adj_matrixs)

    print("[Testing] Test Correctness ...")

    test_algo_correctness(adj_matrixs, edge_feats)

    print("[Benchmark] Start ...")

    benchmark_old(adj_matrixs, edge_feats)
    benchmark_new(adj_matrixs, edge_feats)
    benchmark_numba(adj_matrixs, edge_feats)

if __name__ == "__main__":

    main()
