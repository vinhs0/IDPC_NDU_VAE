import math
import random
from typing import List, Optional

# Assumed project imports
from .Configs import Configs
from .Individual import Individual
from .NodeDepth import NodeDepth

class Population:
    def __init__(self, task):
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.task = task

    def get_population(self) -> List[Individual]:
        return self.population

    def set_population(self, population: List[Individual]):
        self.population = population

    def get_best_individual(self) -> Individual:
        return self.best_individual

    def set_best_individual(self, best_individual: Individual):
        self.best_individual = best_individual

    def init_population(self):
        self.population.clear()

        while len(self.population) < Configs.POPULATION_SIZE:
            i = Individual()
            i.random_init(self.task.adj_domain)
            i.update_fitness(self.task)

            # Check validity (fitness > -MAX_VALUE)
            if i.fitness > -Configs.MAX_VALUE:
                self.population.append(i)

    def eval_population(self):
        for ind in self.population:
            ind.update_fitness(self.task)

    def update_best_individual(self):
        for i in self.population:
            if self.best_individual is None or self.best_individual.fitness < i.fitness:
                self.set_best_individual(i)

    # Edge Permutation Operation (EPO)
    # Moves subtree 'tmp' to be a child of node 'vb'
    def epo(self, va: int, vb: int, tmp: List[NodeDepth], chromosome: List[NodeDepth]):
        node_vb = NodeDepth()
        
        # Find destination node vb info
        for nd in chromosome:
            if nd.node == vb:
                node_vb = NodeDepth(nd)
                break

        # Update depths of the subtree being moved
        # New Depth = Old Depth - Subtree Root Depth + Parent Depth + 1
        dp = tmp[0].depth # Depth of the root of the subtree
        for nd in tmp:
            nd.depth = nd.depth - dp + node_vb.depth + 1

        # Find insertion index (after vb)
        insert_idx = -1
        for i in range(len(chromosome)):
            if chromosome[i].node == vb:
                insert_idx = i
                break
        
        # Insert tmp into chromosome after vb
        # Pythonic list insertion using slicing
        chromosome[insert_idx + 1 : insert_idx + 1] = tmp

    def mutation(self, p: Individual) -> Individual:
        offspring = Individual(p)
        chromosome = offspring.chromosome

        # Pick a random domain on the inter-domain path (excluding source)
        path = offspring.decode(self.task)
        if len(path) > 0:
            path.pop(0) # Remove source domain
        
        if not path:
            return offspring # Safety check if path is empty
            
        va = path[Configs.rd.randint(0, len(path) - 1)] # va: root of pruned subtree
        tmp: List[NodeDepth] = []

        # --- Prune Subtree Logic ---
        # 1. Find start index of va
        start_idx = -1
        for i in range(len(chromosome)):
            if chromosome[i].node == va:
                start_idx = i
                break
        
        # 2. Find the end index of the subtree
        # The subtree consists of the node va and all subsequent nodes with depth > va.depth
        end_idx = start_idx + 1
        va_depth = chromosome[start_idx].depth
        
        while end_idx < len(chromosome):
            if chromosome[end_idx].depth <= va_depth:
                break
            end_idx += 1
            
        # 3. Extract and Remove (Python Slice Optimization)
        # tmp gets deep copies of the nodes
        tmp = [NodeDepth(n) for n in chromosome[start_idx:end_idx]]
        
        # Remove from chromosome
        del chromosome[start_idx:end_idx]

        # --- Select New Parent ---
        # Filter valid parents from task.parent_domain[va]
        # Valid means: the potential parent is NOT currently inside the subtree we just cut out
        # (Though logic dictates we look at the graph definition, the Java code removes nodes present in tmp)
        
        parent_candidates = list(self.task.parent_domain[va])
        
        # Remove nodes that are in the pruned subtree (tmp) from candidates
        # (This prevents attaching the subtree to itself or its own children, though they are already removed from the main tree)
        tmp_node_ids = {nd.node for nd in tmp}
        parent_candidates = [pid for pid in parent_candidates if pid not in tmp_node_ids]

        if not parent_candidates:
            # Fallback if no valid parent found (though uncommon in dense graphs)
            # In standard GA, we might return the original or try again. 
            # Here we just return current state (pruned) or skip EPO.
            # To match Java strictly, we proceed only if list not empty.
             return offspring

        # Choose random vb
        vb = parent_candidates[Configs.rd.randint(0, len(parent_candidates) - 1)]
        
        self.epo(va, vb, tmp, chromosome)

        return offspring

    # Check if vpb (potential parent) is inside the subtree rooted at vri (node being moved)
    # Returns False if vpb is a descendant of vri (Cycle prevention)
    def check(self, vri: int, vpb: int, p1: Individual) -> bool:
        chromosome = p1.chromosome
        
        # Find vri index
        idx = -1
        for i in range(len(chromosome)):
            if chromosome[i].node == vri:
                idx = i
                break
        
        if idx == -1: return True # Should not happen

        depth = chromosome[idx].depth
        
        # Scan forward to check subtree
        for j in range(idx + 1, len(chromosome)):
            # If we hit a node with same or lower depth, we exited the subtree
            if chromosome[j].depth <= depth:
                break
            
            # If we find vpb inside this subtree, it's invalid
            if chromosome[j].node == vpb:
                return False
                
        return True

    # Find the parent of node vri in the tree p
    # In a DFS list, the parent is the nearest preceding node with depth = node.depth - 1
    def find_parent(self, vri: int, p: Individual) -> int:
        rs = 0
        depth = 0
        found = False
        
        # Iterate backwards
        for i in range(len(p.chromosome) - 1, -1, -1):
            nd = p.chromosome[i]
            
            if nd.node == vri:
                found = True
                depth = nd.depth
            
            # Once found, look for the first node with lower depth
            if found and nd.depth < depth:
                rs = nd.node
                break
                
        return rs

    # Edge Copy Operator (ECO) Crossover
    def crossover(self, p1: Individual, p2: Individual) -> Individual:
        offspring = Individual(p1)
        n = self.task.get_number_of_domains()
        
        # Number of nodes to try and exchange
        limit = int(Configs.rd.random() * n / 2 + n / 4)
        
        vr = []
        while len(vr) < limit:
            tmp = Configs.rd.randint(0, n - 1) + 1 # 1 to n
            if tmp > 1 and tmp not in vr:
                vr.append(tmp)
        
        for j in vr:
            # Find who is the parent of j in the second parent (p2)
            p = self.find_parent(j, p2)
            
            # Check if this edge (p -> j) can be replicated in offspring (p1 copy)
            if self.check(j, p, offspring):
                chromosome = offspring.chromosome
                tmp = []

                # --- Prune Subtree j from Offspring ---
                start_idx = -1
                for ii in range(len(chromosome)):
                    if chromosome[ii].node == j:
                        start_idx = ii
                        break
                
                if start_idx != -1:
                    end_idx = start_idx + 1
                    j_depth = chromosome[start_idx].depth
                    
                    while end_idx < len(chromosome):
                        if chromosome[end_idx].depth <= j_depth:
                            break
                        end_idx += 1
                    
                    # Extract subtree
                    tmp = [NodeDepth(node) for node in chromosome[start_idx:end_idx]]
                    
                    # Remove from chromosome
                    del chromosome[start_idx:end_idx]
                    
                    # --- Reattach Subtree ---
                    self.epo(j, p, tmp, chromosome)
                    
        return offspring

    def survival_selection(self):
        # Sort desc (Higher fitness is better)
        # Remember fitness is negative cost, so higher is closer to 0
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Truncate
        if len(self.population) > Configs.POPULATION_SIZE:
            self.population = self.population[:Configs.POPULATION_SIZE]
            
        if self.population:
            self.set_best_individual(self.population[0])