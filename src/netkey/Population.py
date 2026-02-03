import math
from typing import List, Optional

# Assumed project imports
from .Configs import Configs
from .Individual import Individual

class Population:
    def __init__(self, task):
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.task = task

    def init_population(self):
        self.population.clear()
        
        # Based on Java: i.randomInit(task.edges_domain.size())
        # Assuming task.edges_domain is accessible here
        chromosome_len = len(self.task.edges_domain)

        while len(self.population) < Configs.POPULATION_SIZE:
            i = Individual()
            i.random_init(chromosome_len)
            i.update_fitness(self.task)

            # Check validity
            if i.fitness > -Configs.MAX_VALUE:
                self.population.append(i)

    # Getters and Setters
    def get_population(self) -> List[Individual]:
        return self.population

    def set_population(self, population: List[Individual]):
        self.population = population

    def get_best_individual(self) -> Individual:
        return self.best_individual

    def set_best_individual(self, best_individual: Individual):
        self.best_individual = best_individual

    def get_task(self):
        return self.task

    def set_task(self, task):
        self.task = task

    def eval_population(self):
        for ind in self.population:
            ind.update_fitness(self.task)

    def update_best_individual(self):
        for i in self.population:
            if self.best_individual is None or self.best_individual.fitness < i.fitness:
                self.best_individual = i

    def crossover(self, parent1: Individual, parent2: Individual) -> List[Individual]:
        """
        Simulated Binary Crossover (SBX).
        Generates offspring with a distribution property similar to the parents.
        """
        offspring = []
        size_genes = len(parent1.chromosome)
        
        child1_chrom = []
        child2_chrom = []

        for i in range(size_genes):
            # Calculate Beta (Spread Factor)
            cf = 1.0
            u = Configs.rd.random()
            
            if u <= 0.5:
                cf = (2 * u) ** (1.0 / (Configs.mu + 1))
            else:
                cf = (2 * (1 - u)) ** (-1.0 / (Configs.mu + 1))
            
            p1_val = parent1.chromosome[i]
            p2_val = parent2.chromosome[i]

            # Generate Child Values
            v1 = 0.5 * ((1 + cf) * p1_val + (1 - cf) * p2_val)
            v2 = 0.5 * ((1 - cf) * p1_val + (1 + cf) * p2_val)

            # Clamp between 0 and 1
            v1 = max(0.0, min(1.0, v1))
            v2 = max(0.0, min(1.0, v2))

            child1_chrom.append(v1)
            child2_chrom.append(v2)

        offspring.append(Individual(child1_chrom))
        offspring.append(Individual(child2_chrom))
        return offspring

    

    def mutation(self, parent: Individual) -> Individual:
        """
        Polynomial Mutation.
        """
        size_genes = len(parent.chromosome)
        new_chrom = []
        mutation_prob = 1.0 / size_genes

        for i in range(size_genes):
            gene = parent.chromosome[i]
            mutated_val = gene

            if Configs.rd.random() < mutation_prob:
                u = Configs.rd.random()
                delta = 0.0
                
                if u <= 0.5:
                    delta = ((2 * u) ** (1.0 / (1 + Configs.mum))) - 1
                    mutated_val = gene * (delta + 1)
                else:
                    delta = 1 - ((2 * (1 - u)) ** (1.0 / (1 + Configs.mum)))
                    mutated_val = gene + delta * (1 - gene)
            
            # Boundary Handling (Repair Strategy)
            # Logic ported from Java: if out of bounds, randomize within valid range relative to parent
            if mutated_val > 1:
                mutated_val = gene + Configs.rd.random() * (1 - gene)
            elif mutated_val < 0:
                mutated_val = gene * Configs.rd.random()
            
            new_chrom.append(mutated_val)

        return Individual(new_chrom)

    def survival_selection(self):
        # Sort descending (Higher fitness is better)
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # Truncate population to keep size constant
        if len(self.population) > Configs.POPULATION_SIZE:
            self.population = self.population[:Configs.POPULATION_SIZE]
        
        if self.population:
            self.best_individual = self.population[0]