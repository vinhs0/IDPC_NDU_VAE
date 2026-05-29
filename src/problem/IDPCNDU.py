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
        
        # Data structures
        self.domain: List[int] = [] 
        self.distance: List[List[int]] = [] 
        self.list_domain: List[List[int]] = [] 
        self.adj_domain: List[List[int]] = [] 
        self.border_node: List[List[int]] = [] 
        self.parent_domain: List[List[int]] = [] 
        self.domain_edge_count: List[List[int]] = [] # Tracks exact edges between domains
        
        self.adj_node: List[List[int]] = [] 
        self.parent_node: List[List[int]] = [] 
        self.indegree_node: List[int] = [] 
        self.outdegree_node: List[int] = [] 
        self.indegree_domain: List[int] = [] 
        self.outdegree_domain: List[int] = [] 
        
        self.edges_domain: List[Edge] = []

    def get_number_of_nodes(self): return self.number_of_nodes
    def get_number_of_domains(self): return self.number_of_domains
    def get_s(self): return self.s
    def get_t(self): return self.t
    def get_border_node(self): return self.border_node

    def read_data(self, file_path: str):
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

        # Re-init structures with correct sizes
        self.indegree_node = [0] * (self.number_of_nodes + 1)
        self.outdegree_node = [0] * (self.number_of_nodes + 1)
        self.indegree_domain = [0] * (self.number_of_domains + 1)
        self.outdegree_domain = [0] * (self.number_of_domains + 1)
        self.domain = [0] * (self.number_of_nodes + 1)
        
        self.distance = [[Configs.MAX_VALUE] * (self.number_of_nodes + 1) for _ in range(self.number_of_nodes + 1)]
        for i in range(1, self.number_of_nodes + 1):
            self.distance[i][i] = 0
            
        self.list_domain = [[] for _ in range(self.number_of_domains + 1)]
        self.domain_edge_count = [[0] * (self.number_of_domains + 1) for _ in range(self.number_of_domains + 1)]
        
        # Read Domains
        for d in range(1, self.number_of_domains + 1):
            parts = lines[line_idx].split()
            line_idx += 1
            for node_str in parts:
                nid = int(node_str)
                self.domain[nid] = d
                self.list_domain[d].append(nid)
                
        self.start_domain = self.domain[self.s]
        self.target_domain = self.domain[self.t]

        # Init Adjacency Lists
        self.border_node = [[] for _ in range(self.number_of_domains + 1)]
        self.adj_domain = [[] for _ in range(self.number_of_domains + 1)]
        self.parent_domain = [[] for _ in range(self.number_of_domains + 1)]
        self.adj_node = [[] for _ in range(self.number_of_nodes + 1)]
        self.parent_node = [[] for _ in range(self.number_of_nodes + 1)]

        # Read Edges
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

            d_i = self.domain[i]
            d_j = self.domain[j]
            
            if d_i != d_j:
                self.outdegree_domain[d_i] += 1
                self.indegree_domain[d_j] += 1
                
                # Safely track pair connections
                self.domain_edge_count[d_i][d_j] += 1
                
                if d_j not in self.adj_domain[d_i]:
                    self.adj_domain[d_i].append(d_j)
                
                if i not in self.border_node[d_i]:
                    self.border_node[d_i].append(i)
                
                if j not in self.border_node[d_j]:
                    self.border_node[d_j].append(j)
                    
                if d_i not in self.parent_domain[d_j]:
                    self.parent_domain[d_j].append(d_i)

        # FIX 1: Force S and T to be recognized in the border list to avoid Dijkstra disconnects
        if self.s not in self.border_node[self.start_domain]:
            self.border_node[self.start_domain].append(self.s)
        if self.t not in self.border_node[self.target_domain]:
            self.border_node[self.target_domain].append(self.t)

        self.pre_filter_processing()
        self.floyd_warshall()
        
        self.edges_domain = []
        for d in range(1, len(self.adj_domain)):
            for next_d in self.adj_domain[d]:
                self.edges_domain.append(Edge(d, next_d))


    def floyd_warshall(self):
        for d in range(1, self.number_of_domains + 1):
            # FIX 2: Do NOT skip start/target domains. Calculate shortest paths inside them too.
            lst = self.list_domain[d]
            for k in lst:
                for i in lst:
                    for j in lst:
                        if self.distance[i][j] > self.distance[i][k] + self.distance[k][j]:
                            self.distance[i][j] = self.distance[i][k] + self.distance[k][j]

    def update_indegree_domain(self, d: int):
        self.indegree_domain[d] = -1
        for dd in self.adj_domain[d]:
            self.indegree_domain[dd] -= 1
            if d in self.parent_domain[dd]:
                self.parent_domain[dd].remove(d)
            if self.indegree_domain[dd] == 0:
                self.update_indegree_domain(dd)

    def update_outdegree_domain(self, d: int):
        self.outdegree_domain[d] = -1
        for dd in self.parent_domain[d]:
            self.outdegree_domain[dd] -= 1
            if d in self.adj_domain[dd]:
                self.adj_domain[dd].remove(d)
            if self.outdegree_domain[dd] == 0:
                self.update_outdegree_domain(dd)

    def update_indegree(self, node: int):
        self.indegree_node[node] = -1 
        neighbors = list(self.adj_node[node])
        
        for v in neighbors:
            self.indegree_node[v] -= 1
            if node in self.parent_node[v]:
                self.parent_node[v].remove(node)
                
            self.distance[node][v] = Configs.MAX_VALUE
            d_node = self.domain[node]
            d_v = self.domain[v]
            
            if d_node != d_v: 
                if node in self.border_node[d_node]:
                    self.border_node[d_node].remove(node)
                
                # FIX 3a: Sever "Ghost" Domain Edges
                self.domain_edge_count[d_node][d_v] -= 1
                if self.domain_edge_count[d_node][d_v] == 0:
                    if d_v in self.adj_domain[d_node]:
                        self.adj_domain[d_node].remove(d_v)
                    if d_node in self.parent_domain[d_v]:
                        self.parent_domain[d_v].remove(d_node)
                
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
            if node in self.adj_node[v]:
                self.adj_node[v].remove(node)
                
            self.distance[v][node] = Configs.MAX_VALUE
            d_node = self.domain[node]
            d_v = self.domain[v]
            
            if d_node != d_v:
                if node in self.border_node[d_node]:
                    self.border_node[d_node].remove(node)
                
                # FIX 3b: Sever "Ghost" Domain Edges
                self.domain_edge_count[d_v][d_node] -= 1
                if self.domain_edge_count[d_v][d_node] == 0:
                    if d_node in self.adj_domain[d_v]:
                        self.adj_domain[d_v].remove(d_node)
                    if d_v in self.parent_domain[d_node]:
                        self.parent_domain[d_node].remove(d_v)
                
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
            for i in range(1, self.number_of_nodes + 1):
                if i == self.s or i == self.t:
                    continue
                if self.indegree_node[i] == 0:
                    loop_active = True
                    self.update_indegree(i)
                if self.outdegree_node[i] == 0:
                    loop_active = True
                    self.update_outdegree(i)