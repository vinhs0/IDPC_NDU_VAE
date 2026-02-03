import sys
import math
from typing import List, Optional

# Assumed import for configuration
from .Configs import Configs 

# Increase recursion depth for deep DFS operations
sys.setrecursionlimit(10000)

class Individual:
    def __init__(self, source=None):
        """
        Handles constructors:
        1. Individual() -> source=None
        2. Individual(ArrayList<Double> chromosome) -> source=List
        3. Individual(Individual i) -> source=Individual instance
        """
        self.chromosome: List[float] = []
        self.fitness: int = -Configs.MAX_VALUE # Equivalent to Integer.MIN_VALUE

        if source is None:
            pass
        elif isinstance(source, list):
            # Deep copy of the list of doubles
            self.chromosome = list(source)
        elif isinstance(source, Individual):
            # Copy constructor
            self.chromosome = list(source.chromosome)
            self.fitness = source.fitness

    @staticmethod
    def sort_index(data: List[float]) -> List[int]:
        """
        Returns a list of indices that sorts the data in Descending order.
        Equivalent to the Java Comparator logic.
        """
        return sorted(range(len(data)), key=lambda k: data[k], reverse=True)

    def random_init(self, edges_domain_count: int):
        self.chromosome = []
        for _ in range(edges_domain_count):
            self.chromosome.append(Configs.rd.random())

    # --- Cycle Detection Logic ---
    def _is_cyclic_util(self, adj_domain: List[List[int]], v: int, visited: List[bool], parent: int) -> bool:
        visited[v] = True
        
        # Check bounds/existence
        if v < len(adj_domain):
            children = adj_domain[v]
            for c in children:
                if not visited[c]:
                    if self._is_cyclic_util(adj_domain, c, visited, v):
                        return True
                elif c != parent:
                    # If we visit an already visited node that is NOT our direct parent
                    return True
        return False

    def is_cyclic(self, adj_domain: List[List[int]]) -> bool:
        V = len(adj_domain)
        visited = [False] * V
        
        # Java loop starts at 1
        for u in range(1, V):
            if not visited[u]:
                # Start DFS
                if self._is_cyclic_util(adj_domain, u, visited, -1):
                    return True
        return False

    # --- Path Finding (DFS) ---
    def _dfs(self, vis: List[bool], x: int, y: int, stack: List[int], v: List[List[int]], path: List[int]):
        stack.append(x)
        
        if x == y:
            # Path found, copy current stack to path list
            path.extend(stack)
            return

        vis[x] = True
        
        if x < len(v) and len(v[x]) > 0:
            for neighbor in v[x]:
                if not vis[neighbor]:
                    self._dfs(vis, neighbor, y, stack, v, path)
                    # Note: The Java code continues checking other neighbors even if path is found
                    # unless the stack manipulation implicitly handles it.
        
        # Backtrack
        stack.pop()

    def find_path(self, x: int, y: int, stack: List[int], T: List[List[int]], path: List[int]):
        vis = [False] * (len(T) + 1)
        self._dfs(vis, x, y, stack, T, path)

    def decode(self, task) -> List[int]:
        path = []
        
        # Initialize Adjacency List T
        # Assuming adj_domain size covers indices (1-based)
        t_size = len(task.adj_domain)
        T = [[] for _ in range(t_size + 1)]

        C = [] # Visited domains
        edges_idx = self.sort_index(self.chromosome)
        
        edge_num = 0
        
        for edge_i in edges_idx:
            e = task.edges_domain[edge_i]
            u = e.node1
            v = e.node2
            
            # Ensure T list size logic (Python dynamic sizing safety)
            while len(T) <= u: T.append([])
            while len(T) <= v: T.append([])
            
            T[u].append(v)
            # T[v].append(u) # Commented out in Java

            if self.is_cyclic(T):
                # Backtrack: Remove the edge just added
                T[u].pop()
                continue
            
            if u not in C: C.append(u)
            if v not in C: C.append(v)
            
            # Added increment to make the break condition reachable
            edge_num += 1 
            
            if edge_num == len(task.adj_domain) - 2:
                break
        
        # Find path from Start (1) to End (Size - 1)
        target = len(task.adj_domain) - 1
        self.find_path(1, target, [], T, path)
        
        return path

    # --- Dijkstra Helpers ---
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
        # Initialize distance matrix
        distance = [[Configs.MAX_VALUE] * (num_nodes + 1) for _ in range(num_nodes + 1)]
        
        for i in range(1, num_nodes + 1):
            distance[i][i] = 0
            
        # Iterate through the sequence of domains in the path
        for i in range(len(path)):
            current_domain = path[i]
            
            # Check bounds safety
            if current_domain >= len(task.border_node): continue

            list_borders_this = task.border_node[current_domain]
            
            # 1. Edges within the current domain
            for j in list_borders_this:
                for k in list_borders_this:
                    if task.distance[j][k] != Configs.MAX_VALUE:
                        distance[j][k] = task.distance[j][k]
            
            # 2. Edges to subsequent domains (Shortcut paths)
            for j in range(i + 1, len(path)):
                next_domain = path[j]
                if next_domain >= len(task.border_node): continue
                
                list_borders_that = task.border_node[next_domain]
                
                for x in list_borders_this:
                    for y in list_borders_that:
                        if task.distance[x][y] != Configs.MAX_VALUE:
                            distance[x][y] = task.distance[x][y]
                            
        return distance

    def dijkstra(self, task, list_nodes: List[int], distance_matrix: List[List[int]]) -> int:
        num_nodes = task.get_number_of_nodes()
        dist = [Configs.MAX_VALUE] * (num_nodes + 1)
        visited = [False] * (num_nodes + 1)
        
        # NOTE: The original Java code contained: `distance = task.distance;`
        # This overwrites the passed subgraph with the global graph, rendering buildGraph useless.
        # We use the passed `distance_matrix` (the subgraph) to respect the method signature.
        distance = distance_matrix 

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
        # 1. Decode chromosome to find the sequence of domains
        path_domains = self.decode(task)
        
        # 2. Collect all border nodes belonging to these domains
        list_nodes = []
        for d in path_domains:
            if d < len(task.border_node):
                list_nodes.extend(task.border_node[d])
        
        # 3. Build the virtual graph restricted to these domains
        distance_matrix = self.build_graph(task, path_domains, list_nodes)
        
        # 4. Calculate cost
        cost = self.dijkstra(task, list_nodes, distance_matrix)
        
        self.fitness = -cost