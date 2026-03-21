import copy
import math
import numpy as np
from typing import List

from .VAEConfigs import Configs
from ga.Edge import Edge
from vae.NodeDepth import NodeDepth 

class Individual:
    def __init__(self, source=None):
        self.chromosome = [] #: List[NodeDepth] 
        self.fitness = -float('inf') 
        
        # Stats
        self.total_domain: int = 0
        self.total_node: int = 0
        self.total_edge: int = 0
        self.domain: int = 0
        self.node: int = 0
        self.edge: int = 0

        if source is None:
            pass
        elif isinstance(source, list):
            self.chromosome = [NodeDepth(n.node, n.depth) for n in source]
        elif isinstance(source, Individual):
            self.chromosome = [NodeDepth(n.node, n.depth) for n in source.chromosome]
            self.fitness = source.fitness
    # Biến data thành graph (1 dạng ctdl nào đó có graph)
    # Ở đây ta sẽ đổi từ cái này thành node embeddings với Node2Vec (dùng cái khác nếu cái này tệ)
    # Chuyển method này về 1 chỗ khác (Embedder.py)
    # def convert_NDList2Graph(self):
    #     """
    #         Decode NodeDepth thành 2 thứ: Node feature matrix và COO list.

    #         x: List chứa [Node ID (tên) và Node depth]
    #         edge_sources: Tập các cạnh được biểu diễn là [i,j] với i -> j 
    #         edge_targets: ngược lại
    #     """
    #     x = []
    #     edge_sources = []
    #     edge_targets = []
    #     # Chromosome là 1 list NodeDepth
    #     for i, nd in enumerate(self.chromosome):
    #         x.append([float(nd.node), float(nd.depth)])
    #         if i > 0:
    #             for j in range(i - 1, -1, -1):
    #                 # tại mỗi chromosome, nếu depth của nó = depth node htai - 1 => 2 node này kề nhau
    #                 if self.chromosome[j].depth == nd.depth - 1:
    #                     edge_sources.extend([j, i])
    #                     edge_targets.extend([i, j])
    #                     break
        
    #     # Kêt quả là một Node feature matrix và COO format
    #     return np.array(x, dtype=np.float32), np.array([edge_sources, edge_targets], dtype=np.int64)

    def random_init(self, adj_domain: List[List[int]]):
        st = self.prim_rst(adj_domain)
        self.chromosome = self.encode(st)
        self.total_domain = len(adj_domain) - 1 if adj_domain[0] is None else len(adj_domain)

    def get_chromosome(self):
        return self.chromosome

    def set_chromosome(self, chromosome):
        self.chromosome = chromosome

    def _dfs_util(self, v: int, visited: List[bool], depth: List[int], 
                  depth_of_v: int, rs: List[NodeDepth], t: List[List[int]]):
        visited[v] = True
        depth[v] = depth_of_v
        rs.append(NodeDepth(v, depth[v]))

        if v < len(t):
            adj = t[v]
            for u in adj:
                if not visited[u]:
                    self._dfs_util(u, visited, depth, depth_of_v + 1, rs, t)

    def dfs(self, t: List[List[int]]) -> List[NodeDepth]:
        size = len(t)
        visited = [False] * size
        depth = [0] * size
        rs = []

        if size > 1:
            depth[1] = 0
            self._dfs_util(1, visited, depth, depth[1], rs, t)
        
        return rs

    def encode(self, t: List[List[int]]) -> List[NodeDepth]:
        return self.dfs(t)

    def prim_rst(self, adj_domain: List[List[int]]) -> List[List[int]]:
        size = len(adj_domain)
        # Match Java's T.add(new ArrayList()) structure (1-based indexing)
        t = [[] for _ in range(size + 1)] 
        
        c = [1] 
        a: List[Edge] = [] 

        start_neighbors = adj_domain[1] if len(adj_domain) > 1 else []
        for v in start_neighbors:
            a.append(Edge(1, v))

        target_size = len(adj_domain) - 1 
        
        while len(c) != target_size and len(a) > 0:
            rand_index = Configs.rd.randint(0, len(a) - 1)
            e = a.pop(rand_index)
            
            u = e._node1
            v = e._node2

            if v not in c:
                t[u].append(v)
                c.append(v)
                
                if v < len(adj_domain):
                    for w in adj_domain[v]:
                        if w not in c:
                            a.append(Edge(v, w))
        return t

    # ---------------------------------------------------------
    # Decoding & Path Reconstruction
    # ---------------------------------------------------------
    def decode(self, task) -> List[int]:
        path = []
        if not self.chromosome: return []
        
        u = [NodeDepth(x.node, x.depth) for x in self.chromosome]
        tree: List[NodeDepth] = []
        visited = [0] * len(u) 
        index_list = [] # Translating Java's "index" ArrayList
        
        # Init root
        tree.append(u[0])
        visited[0] = 1
        index_list.append(0)

        # Safety breakout to avoid infinite loops if VAE generates bad chromosome
        max_iterations = len(u) * 5
        loops = 0

        while sum(visited) != len(u) and loops < max_iterations:
            loops += 1
            added_this_round = False
            
            for j in range(1, len(u)):
                if visited[j] == 1:
                    continue
                
                x = u[j]
                if x.node > task.get_number_of_domains():
                    visited[j] = 1
                    continue
                
                root_node = -1
                for k in range(len(tree) - 1, -1, -1):
                    y = tree[k]
                    # Java Logic: Depth must be strictly less, and nodes must be connected
                    if y.depth < x.depth and x.node in task.adj_domain[y.node]:
                        if root_node == -1:
                            root_node = k
                            continue
                        # Java Logic: pick the root that is closest in the original chromosome sequence
                        if abs(index_list[root_node] - j) > abs(index_list[k] - j):
                            root_node = k

                if root_node != -1:
                    # Find right-most boundary of this subtree
                    idx = root_node + 1
                    while idx < len(tree):
                        if tree[idx].depth <= tree[root_node].depth:
                            break
                        idx += 1
                    
                    # Insert
                    if idx < len(tree):
                        tree.insert(idx, NodeDepth(x.node, tree[root_node].depth + 1))
                        index_list.insert(idx, j)
                    else:
                        tree.append(NodeDepth(x.node, tree[root_node].depth + 1))
                        index_list.append(j)
                        
                    visited[j] = 1
                    added_this_round = True
                
                if visited[j] == 0:
                    u[j].depth += 1
                    
            if not added_this_round and sum(visited) != len(u):
                break # VAE broke the tree, exit to prevent infinite loop
                
        # Path Extraction
        p = NodeDepth(0, 0)
        idx_target = -1
        target_domain = task.get_number_of_domains()
        
        for i in range(len(tree) - 1, -1, -1):
            if tree[i].node == target_domain:
                path.append(target_domain)
                p = tree[i]
                idx_target = i
                break
        
        if idx_target == -1: return [] # Failsafe

        for j in range(idx_target - 1, -1, -1):
            if tree[j].depth < p.depth:
                p = tree[j]
                path.append(p.node)

        path.reverse()
        return path

    # ---------------------------------------------------------
    # Graph & Dijkstra Costs
    # ---------------------------------------------------------
    def _min_distance(self, dist: List[int], visited: List[bool], list_nodes: List[int]) -> int:
        min_val = Configs.MAX_VALUE
        min_index = -1
        
        for v in list_nodes:
            if not visited[v] and dist[v] < min_val:
                min_index = v
                min_val = dist[v]
        
        return min_index

    def build_graph(self, task, path: List[int], list_nodes: List[int]) -> List[List[int]]:
        num_nodes = task.get_number_of_nodes()
        distance = [[Configs.MAX_VALUE] * (num_nodes + 1) for _ in range(num_nodes + 1)]
        
        for i in range(1, num_nodes + 1):
            distance[i][i] = 0
            
        edge_count = 0
        node_count = 0
        
        for i in range(len(path)):
            current_domain = path[i]
            list_borders_this = task.get_border_node()[current_domain]
            node_count += len(list_borders_this)
            
            for j in list_borders_this:
                for k in list_borders_this:
                    if task.distance[j][k] != Configs.MAX_VALUE:
                        distance[j][k] = task.distance[j][k]
                        edge_count += 1
            
            for j in range(i + 1, len(path)):
                next_domain = path[j]
                list_borders_that = task.get_border_node()[next_domain]
                
                for x in list_borders_this:
                    for y in list_borders_that:
                        if task.distance[x][y] != Configs.MAX_VALUE:
                            distance[x][y] = task.distance[x][y]

        self.total_edge = task.number_of_edges
        self.domain = len(path)
        self.node = node_count
        self.edge = edge_count
        
        return distance

    def dijkstra(self, task, list_nodes: List[int], distance_matrix: List[List[int]]) -> int:
        num_nodes = task.get_number_of_nodes()
        dist = [Configs.MAX_VALUE] * (num_nodes + 1)
        visited = [False] * (num_nodes + 1)
        
        # VERY IMPORTANT JAVA QUIRK:
        # In the original Java code, it accepted `distance` as an argument, 
        # but immediately overwrote it with `distance = task.distance;`.
        # We mirror this behavior here to ensure costs are exactly identical.
        adj = task.distance 
        
        s_node = task.get_s()
        t_node = task.get_t()
        
        dist[s_node] = 0

        while True:
            u = self._min_distance(dist, visited, list_nodes)
            
            if u == -1:
                return Configs.MAX_VALUE
            
            visited[u] = True
            
            if u == t_node:
                break
            
            for v in list_nodes:
                if not visited[v] and u != v:
                    if adj[u][v] != Configs.MAX_VALUE:
                        if dist[v] > dist[u] + adj[u][v]:
                            dist[v] = dist[u] + adj[u][v]
                            
        return dist[t_node]

    def update_fitness(self, task):
        path = self.decode(task)
        
        if not path or len(path) == 0:
            self.fitness = -Configs.MAX_VALUE
            return
            
        list_nodes = []
        for d in path:
            # Safely append all border nodes associated with the domains in our path
            if d < len(task.border_node):
                list_nodes.extend(task.border_node[d])
                
        if not list_nodes:
            self.fitness = -Configs.MAX_VALUE
            return
            
        distance_matrix = self.build_graph(task, path, list_nodes)
        cost = self.dijkstra(task, list_nodes, distance_matrix)
        
        self.total_domain = task.get_number_of_domains()
        self.total_node = task.get_number_of_nodes()
        
        self.fitness = -cost