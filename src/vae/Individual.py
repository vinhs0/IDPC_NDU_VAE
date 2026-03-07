import copy
import math
import numpy as np
from typing import List, Dict

# Assumed imports based on project structure
from ga.Configs import Configs
from ga.Edge import Edge
from ga.NodeDepth import NodeDepth 

class Individual:
    def __init__(self, source=None):
        self.chromosome: List[NodeDepth] = []
        self.fitness: int = -float('inf') 
        
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

    # ---------------------------------------------------------
    # NEW METHOD: Extract graph data for GNN VAE
    # ---------------------------------------------------------
    def get_graph_data(self):
        """
        Reconstructs the graph (tree) from the DFS chromosome.
        Returns:
            x (np.array): Node features [Node ID, Depth]
            edge_index (np.array): Graph connectivity [2, num_edges]
        """
        x = []
        edge_sources = []
        edge_targets = []
        
        for i, nd in enumerate(self.chromosome):
            # Node feature (Scale depth slightly or keep raw)
            x.append([float(nd.node), float(nd.depth)])
            
            # Find parent to create an edge based on DFS structure
            if i > 0:
                # Iterate backward to find the immediate parent
                for j in range(i - 1, -1, -1):
                    if self.chromosome[j].depth == nd.depth - 1:
                        # Undirected edge for the GNN Message Passing
                        edge_sources.append(j)
                        edge_targets.append(i)
                        edge_sources.append(i)
                        edge_targets.append(j)
                        break
                        
        x_array = np.array(x, dtype=np.float32)
        edge_index_array = np.array([edge_sources, edge_targets], dtype=np.int64)
        
        return x_array, edge_index_array
    # ---------------------------------------------------------

    def random_init(self, adj_domain: List[List[int]]):
        st = self.prim_rst(adj_domain)
        self.chromosome = self.encode(st)
        self.total_domain = len(adj_domain) - 1 if adj_domain[0] is None else len(adj_domain)

    def get_chromosome(self) -> List[NodeDepth]:
        return self.chromosome

    def set_chromosome(self, chromosome: List[NodeDepth]):
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
        t = [[] for _ in range(size + 1)] 
        
        c = [] 
        a: List[Edge] = [] 

        c.append(1)
        start_neighbors = adj_domain[1] if len(adj_domain) > 1 else []
        
        for v in start_neighbors:
            a.append(Edge(1, v))

        target_size = len(adj_domain) - 1 
        
        while len(c) != target_size and len(a) > 0:
            rand_index = Configs.rd.randint(0, len(a) - 1)
            e = a[rand_index]
            
            u = e._node1
            v = e._node2
            
            a.pop(rand_index)

            if v not in c:
                while len(t) <= u: t.append([])
                t[u].append(v)
                
                c.append(v)
                
                if v < len(adj_domain):
                    for w in adj_domain[v]:
                        if w not in c:
                            a.append(Edge(v, w))
        return t

    def decode(self, task) -> List[int]:
        path = []
        u = [NodeDepth(x.node, x.depth) for x in self.chromosome]
        
        tree: List[NodeDepth] = []
        visited = [0] * len(self.chromosome)
        
        tree.append(u[0])
        visited[0] = 1

        while sum(visited) != len(u):
            for j in range(1, len(u)):
                if visited[j] == 1:
                    continue
                
                x = u[j]
                
                if x.node > task.get_number_of_domains():
                    visited[j] = 1
                    continue
                
                attached = False
                for k in range(len(tree) - 1, -1, -1):
                    y = tree[k]
                    if y.depth < x.depth and x.node in task.adj_domain[y.node]:
                        new_node = NodeDepth(x.node, y.depth + 1)
                        tree.insert(k + 1, new_node)
                        visited[j] = 1
                        attached = True
                        break
                if not attached:
                    u[j].depth += 1

        p = NodeDepth(0, 0)
        idx = -1
        
        for i in range(len(tree) - 1, -1, -1):
            if tree[i].node == task.get_number_of_domains():
                path.append(task.get_number_of_domains())
                p = tree[i]
                idx = i
                break
        
        for j in range(idx - 1, -1, -1):
            if tree[j].depth < p.depth:
                p = tree[j]
                path.append(p.node)

        path.reverse()
        return path

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

    def dijkstra(self, task, list_nodes: List[int], distance: List[List[int]]) -> int:
        num_nodes = task.get_number_of_nodes()
        dist = [Configs.MAX_VALUE] * (num_nodes + 1)
        visited = [False] * (num_nodes + 1)
        
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
                    if distance[u][v] != Configs.MAX_VALUE:
                        if dist[v] > dist[u] + distance[u][v]:
                            dist[v] = dist[u] + distance[u][v]
                            
        return dist[t_node]

    def update_fitness(self, task):
        path = self.decode(task)
        
        list_nodes = []
        for d in path:
            list_nodes.extend(task.border_node[d])
            
        distance_matrix = self.build_graph(task, path, list_nodes)
        cost = self.dijkstra(task, list_nodes, distance_matrix)
        
        self.total_domain = task.get_number_of_domains()
        self.total_node = task.get_number_of_nodes()
        
        self.fitness = -cost