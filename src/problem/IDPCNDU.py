import sys
import math
from typing import List, Optional

# Project imports
from ga.Configs import Configs
from .Edge import Edge

# Increase recursion depth for deep graph pruning
sys.setrecursionlimit(20000)

class IDPCNDU:
    def __init__(self):
        self.number_of_nodes: int = 0
        self.number_of_domains: int = 0
        self.number_of_edges: int = 0
        self.s: int = 0  # source
        self.t: int = 0  # destination
        self.start_domain: int = 0
        self.target_domain: int = 0
        
        # Data structures (Initialized to None or empty lists later)
        # Using 1-based indexing means lists will be size N+1
        self.domain: List[int] = [] 
        self.distance: List[List[int]] = [] 
        self.list_domain: List[List[int]] = [] 
        self.adj_domain: List[List[int]] = [] 
        self.border_node: List[List[int]] = [] 
        self.parent_domain: List[List[int]] = [] 
        
        self.adj_node: List[List[int]] = [] 
        self.parent_node: List[List[int]] = [] 
        self.indegree_node: List[int] = [] 
        self.outdegree_node: List[int] = [] 
        self.indegree_domain: List[int] = [] 
        self.outdegree_domain: List[int] = [] 
        
        self.edges_domain: List[Edge] = []

    # Getters/Setters are Python properties or direct access
    def get_number_of_nodes(self): return self.number_of_nodes
    def get_number_of_domains(self): return self.number_of_domains
    def get_s(self): return self.s
    def get_t(self): return self.t
    def get_border_node(self): return self.border_node
    # ... In Python, we typically access attributes directly (e.g., task.s, task.domain)

    def read_data(self, file_path: str):
        # --- Reliable Parsing Strategy ---
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        lines = [l.strip() for l in lines if l.strip()]
        line_idx = 0
        
        # Header
        header = lines[line_idx].split()
        self.number_of_nodes = int(header[0])
        self.number_of_domains = int(header[1])
        line_idx += 1

        endpoints = lines[line_idx].split()
        self.s = int(endpoints[0])
        self.t = int(endpoints[1])
        line_idx += 1

        # Re-init structures
        self.indegree_node = [0] * (self.number_of_nodes + 1)
        self.outdegree_node = [0] * (self.number_of_nodes + 1)
        self.indegree_domain = [0] * (self.number_of_domains + 1)
        self.outdegree_domain = [0] * (self.number_of_domains + 1)
        self.domain = [0] * (self.number_of_nodes + 1)
        
        self.distance = [[Configs.MAX_VALUE] * (self.number_of_nodes + 1) for _ in range(self.number_of_nodes + 1)]
        for i in range(1, self.number_of_nodes + 1):
            self.distance[i][i] = 0
            
        self.list_domain = [[] for _ in range(self.number_of_domains + 1)]
        
        # FIX: Read exactly `self.number_of_domains` lines sequentially
        for d in range(1, self.number_of_domains + 1):
            parts = lines[line_idx].split()
            line_idx += 1
            for node_str in parts:
                nid = int(node_str)
                self.domain[nid] = d
                self.list_domain[d].append(nid)
                
        # FIX: Dynamically register Start and Target Domains
        self.start_domain = self.domain[self.s]
        self.target_domain = self.domain[self.t]

        # Init Adjacency Lists
        self.border_node = [[] for _ in range(self.number_of_domains + 1)]
        self.adj_domain = [[] for _ in range(self.number_of_domains + 1)]
        self.parent_domain = [[] for _ in range(self.number_of_domains + 1)]
        self.adj_node = [[] for _ in range(self.number_of_nodes + 1)]
        self.parent_node = [[] for _ in range(self.number_of_nodes + 1)]

        # Read Edges (Remaining lines)
        while line_idx < len(lines):
            edge_parts = lines[line_idx].split()
            line_idx += 1
            if len(edge_parts) < 3: continue
            
            i = int(edge_parts[0])
            j = int(edge_parts[1])
            w = int(edge_parts[2])

            if j not in self.adj_node[i]: self.adj_node[i].append(j)
            if i not in self.parent_node[j]: self.parent_node[j].append(i)

            self.distance[i][j] = min(self.distance[i][j], w)
            self.number_of_edges += 1
            self.outdegree_node[i] += 1
            self.indegree_node[j] += 1

            # Update Domain Graph
            d_i = self.domain[i]
            d_j = self.domain[j]
            
            if d_i != d_j:
                self.outdegree_domain[d_i] += 1
                self.indegree_domain[d_j] += 1
                
                if d_j not in self.adj_domain[d_i]:
                    self.adj_domain[d_i].append(d_j)
                
                if i not in self.border_node[d_i]:
                    self.border_node[d_i].append(i)
                
                if j not in self.border_node[d_j]:
                    self.border_node[d_j].append(j)
                    
                if d_i not in self.parent_domain[d_j]:
                    self.parent_domain[d_j].append(d_i)

        self.pre_filter_processing()
        self.floyd_warshall()
        
        self.edges_domain = []
        for d in range(1, len(self.adj_domain)):
            for next_d in self.adj_domain[d]:
                self.edges_domain.append(Edge(d, next_d))


    def floyd_warshall(self):
        for d in range(1, self.number_of_domains + 1):
            if d == self.start_domain or d == self.target_domain:
                continue
            
            lst = self.list_domain[d]
            # O(N^3) within the domain
            for k in lst:
                for i in lst:
                    for j in lst:
                        if self.distance[i][j] > self.distance[i][k] + self.distance[k][j]:
                            self.distance[i][j] = self.distance[i][k] + self.distance[k][j]

    def update_indegree_domain(self, d: int):
        self.indegree_domain[d] = -1
        for dd in self.adj_domain[d]:
            self.indegree_domain[dd] -= 1
            
            # Remove d from parent_domain[dd]
            if d in self.parent_domain[dd]:
                self.parent_domain[dd].remove(d)
            
            if self.indegree_domain[dd] == 0:
                self.update_indegree_domain(dd)

    def update_outdegree_domain(self, d: int):
        self.outdegree_domain[d] = -1
        for dd in self.parent_domain[d]:
            self.outdegree_domain[dd] -= 1
            
            # Remove d from adj_domain[dd]
            if d in self.adj_domain[dd]:
                self.adj_domain[dd].remove(d)
                
            if self.outdegree_domain[dd] == 0:
                self.update_outdegree_domain(dd)

    def update_indegree(self, node: int):
        self.indegree_node[node] = -1 # Mark as removed
        
        # Iterate over copy since we might recurse
        neighbors = list(self.adj_node[node])
        
        for v in neighbors:
            self.indegree_node[v] -= 1
            
            # Remove node from parent_node[v]
            if node in self.parent_node[v]:
                self.parent_node[v].remove(node)
                
            self.distance[node][v] = Configs.MAX_VALUE
            
            d_node = self.domain[node]
            d_v = self.domain[v]
            
            if d_node != d_v: # If they are in different domains (Border Logic)
                if node in self.border_node[d_node]:
                    self.border_node[d_node].remove(node)
                
                self.outdegree_domain[d_node] -= 1
                if self.outdegree_domain[d_node] == 0:
                    self.update_outdegree_domain(d_node)
                    
                self.indegree_domain[d_v] -= 1
                if self.indegree_domain[d_v] == 0:
                    self.update_indegree_domain(d_v)
            
            if self.indegree_node[v] == 0:
                self.update_indegree(v)

    def update_outdegree(self, node: int):
        self.outdegree_node[node] = -1
        
        parents = list(self.parent_node[node])
        
        for v in parents:
            self.outdegree_node[v] -= 1
            
            # Remove node from adj_node[v]
            if node in self.adj_node[v]:
                self.adj_node[v].remove(node)
                
            self.distance[v][node] = Configs.MAX_VALUE
            
            d_node = self.domain[node]
            d_v = self.domain[v]
            
            if d_node != d_v:
                if node in self.border_node[d_node]:
                    self.border_node[d_node].remove(node)
                
                self.indegree_domain[d_node] -= 1
                if self.indegree_domain[d_node] == 0:
                    self.update_indegree_domain(d_node)
                
                self.outdegree_domain[d_v] -= 1
                if self.outdegree_domain[d_v] == 0:
                    self.update_outdegree_domain(d_v)
            
            if self.outdegree_node[v] == 0: 
                self.update_outdegree(v)

    def pre_filter_processing(self):
        loop_active = True
        while loop_active:
            loop_active = False 
            
            # Safely loop through all nodes
            for i in range(1, self.number_of_nodes + 1):
                
                # Protect Source and Target from being deleted
                if i == self.s or i == self.t:
                    continue
                    
                if self.indegree_node[i] == 0:
                    loop_active = True
                    self.update_indegree(i)
                
                if self.outdegree_node[i] == 0:
                    loop_active = True
                    self.update_outdegree(i)