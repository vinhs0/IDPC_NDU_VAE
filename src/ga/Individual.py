import copy
import math
from typing import List, Dict

# Assumed imports based on project structure
from .Configs import Configs
from .Edge import Edge
# You will need a NodeDepth class similar to the Edge class refactor
from .NodeDepth import NodeDepth 

class Individual:
    def __init__(self, source=None):
        """
        Handles three Java constructors:
        1. Individual() -> source=None
        2. Individual(ArrayList<NodeDepth> chromosome) -> source=List
        3. Individual(Individual i) -> source=Individual instance
        """
        self.chromosome: List[NodeDepth] = []
        self.fitness: int = -float('inf') # Equivalent to Integer.MIN_VALUE
        
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

        # In Python, check bounds if T is 0-indexed or 1-indexed. 
        # Assuming T follows Java's 1-based logic mapping:
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

        # Assuming start node is 1 (based on Java code)
        if size > 1:
            depth[1] = 0
            self._dfs_util(1, visited, depth, depth[1], rs, t)
        
        return rs

    def encode(self, t: List[List[int]]) -> List[NodeDepth]:
        return self.dfs(t)

    def prim_rst(self, adj_domain: List[List[int]]) -> List[List[int]]:
        # Initialize Tree structure (adjacency list)
        # +1 to handle 1-based indexing common in these algo problems
        size = len(adj_domain)
        t = [[] for _ in range(size + 1)] 
        
        c = [] # Visited nodes
        a: List[Edge] = [] # Candidate edges

        # Init (Start at 1)
        c.append(1)
        # adj_domain likely 1-based, if 0-based adjust access to adj_domain[1]
        start_neighbors = adj_domain[1] if len(adj_domain) > 1 else []
        
        for v in start_neighbors:
            a.append(Edge(1, v))

        # While tree doesn't contain all domains
        # (Assuming adj_domain size includes index 0 as dummy)
        target_size = len(adj_domain) - 1 
        
        while len(c) != target_size and len(a) > 0:
            # Random pick
            rand_index = Configs.rd.randint(0, len(a) - 1)
            e = a[rand_index]
            
            u = e._node1
            v = e._node2
            
            # Efficient remove (swap with last and pop is O(1), but standard pop is O(N))
            a.pop(rand_index)

            if v not in c:
                # Add to tree (ensure list exists)
                while len(t) <= u: t.append([])
                t[u].append(v)
                
                c.append(v)
                
                # Add new neighbors
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
                
                # Check for invalid node ID
                if x.node > task.get_number_of_domains():
                    visited[j] = 1
                    continue
                
                # Try to attach x to the existing tree
                attached = False
                for k in range(len(tree) - 1, -1, -1):
                    y = tree[k]
                    # Logic: y has lower depth AND y is adjacent to x in domain graph
                    if y.depth < x.depth and x.node in task.adj_domain[y.node]:
                        # Insert x into tree after y (or appropriately logic-wise)
                        # Java: tree.add(k+1, new NodeDepth(x.node, y.depth+1))
                        new_node = NodeDepth(x.node, y.depth + 1)
                        tree.insert(k + 1, new_node)
                        visited[j] = 1
                        attached = True
                        break
                if not attached:
                    # Increment depth and try again next loop?
                    # Java: u.get(j).setDepth(u.get(j).getDepth()+1);
                    u[j].depth += 1

        # Reconstruct path from tree
        # Find destination domain
        p = NodeDepth(0, 0)
        idx = -1
        
        for i in range(len(tree) - 1, -1, -1):
            if tree[i].node == task.get_number_of_domains():
                path.append(task.get_number_of_domains())
                p = tree[i]
                idx = i
                break
        
        # Trace back
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
        # Initialize distance matrix with MAX_VALUE
        distance = [[Configs.MAX_VALUE] * (num_nodes + 1) for _ in range(num_nodes + 1)]
        
        for i in range(1, num_nodes + 1):
            distance[i][i] = 0
            
        edge_count = 0
        node_count = 0
        
        for i in range(len(path)):
            current_domain = path[i]
            # Get border nodes for this domain
            list_borders_this = task.get_border_node()[current_domain]
            node_count += len(list_borders_this)
            
            # Build edges within the domain
            for j in list_borders_this:
                for k in list_borders_this:
                    if task.distance[j][k] != Configs.MAX_VALUE:
                        distance[j][k] = task.distance[j][k]
                        edge_count += 1
            
            # Build shortcut paths (edges between domains)
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
        
        # NOTE: The original Java code has this line:
        # distance = task.distance;
        # This overwrites the subgraph 'distance' passed in with the global graph.
        # This renders build_graph irrelevant for the cost calculation.
        # Uncomment the line below if you want EXACT logic parity with the Java snippet provided.
        # distance = task.distance 

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
                     # Check if edge exists
                    if distance[u][v] != Configs.MAX_VALUE:
                        if dist[v] > dist[u] + distance[u][v]:
                            dist[v] = dist[u] + distance[u][v]
                            
        return dist[t_node]

    def update_fitness(self, task):
        path = self.decode(task)
        
        list_nodes = []
        for d in path:
            # Add all border nodes of this domain
            list_nodes.extend(task.border_node[d])
            
        distance_matrix = self.build_graph(task, path, list_nodes)
        
        # Calculate Dijkstra on the specific subgraph (list_nodes)
        cost = self.dijkstra(task, list_nodes, distance_matrix)
        
        self.total_domain = task.get_number_of_domains()
        self.total_node = task.get_number_of_nodes()
        
        self.fitness = -cost