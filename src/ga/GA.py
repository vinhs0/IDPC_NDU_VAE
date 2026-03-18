import os
import time
import math
from typing import List

# Assuming these are in your project structure
from .Configs import Configs
from .Population import Population
from .Individual import Individual

class GA:
    def __init__(self, task, output_path: str, file_name: str):
        self.task = task
        self.output_path = output_path
        self.file_name = file_name

    def save(self, seed: int, pop: Population, t1: float, t2: float):
        out_file_name_opt = f"{self.file_name}_seed({seed}).opt"
        full_path = os.path.join(self.output_path, out_file_name_opt)

        try:
            with open(full_path, "w") as fw_opt:
                fw_opt.write(f"Filename: {self.file_name}\n")
                fw_opt.write(f"Seed: {seed}\n")
                # Note: Preserving the negative logic from Java code (-fitness)
                fw_opt.write(f"Fitness: {-pop.get_best_individual().fitness}\n")

                duration_sec = t2 - t1
                
                # Manual time formatting to match Java's TimeUnit logic
                hours = int(duration_sec // 3600)
                minutes = int((duration_sec % 3600) // 60)
                seconds = int(duration_sec % 60)
                milliseconds = int((duration_sec * 1000) % 1000)

                time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
                
                fw_opt.write(f"Time: {time_str}\n")
                fw_opt.flush()
        except IOError as e:
            print(f"Error saving file: {e}")

    def run(self, seed: int) -> Individual:
        t1 = time.time()
        
        population = Population(self.task)
        population.init_population() # init and update fitness
        population.update_best_individual()

        generation = 0
        out_file_name_gen = f"{self.file_name}_seed({seed}).gen"
        full_path = os.path.join(self.output_path, out_file_name_gen)
        
        print(out_file_name_gen)

        with open(full_path, "w") as fw_gen:
            fw_gen.write(f"Generations {self.file_name}\n")

            while generation < Configs.MAX_GENERATIONS:
                best_fitness = -population.get_best_individual().fitness
                #print fitness
                print(f"Gen {generation}, fitness: {best_fitness}")
                fw_gen.write(f"{generation} {best_fitness}\n")
                fw_gen.flush()

                # Create offspring (subset)
                offspring = self.reproduction(population.get_population())

                # Combine old population vs new population
                imi_pop = []
                imi_pop.extend(offspring)
                imi_pop.extend(population.get_population())

                # Select individuals for next generation
                population.set_population(imi_pop)
                population.survival_selection()
                generation += 1

        t2 = time.time()
        self.save(seed, population, t1, t2)
        return population.get_best_individual()

    # For getting data of domain, node, edge when decoding
    def run2(self, seed: int) -> Individual:
        population = Population(self.task)
        population.init_population()
        population.update_best_individual()

        generation = 0
        out_file_name_opt = f"{self.file_name}_seed({seed}).opt"
        full_path = os.path.join(self.output_path, out_file_name_opt)

        with open(full_path, "w") as fw_gen:
            fw_gen.write("Gen\tTotal_domain\tDomain\tTotal_node\tNode\tTotal_edge\tEdge\n")

            while generation < Configs.MAX_GENERATIONS:
                # #print fitness
                # print(f"Gen {generation}, fitness: {population.get_best_individual().fitness}")
                b = population.get_best_individual()
                
                # Assuming Individual has these attributes exposed
                line = (f"{generation}\t{b.total_domain}\t{b.domain}\t"
                        f"{b.total_node}\t{b.node}\t{b.total_edge}\t{b.edge}\n")
                
                fw_gen.write(line)
                fw_gen.flush()

                offspring = self.reproduction(population.get_population())

                imi_pop = []
                imi_pop.extend(offspring)
                imi_pop.extend(population.get_population())

                population.set_population(imi_pop)
                population.survival_selection()
                generation += 1

        return population.get_best_individual()

    # Tournament selection with k = 2
    def select_parent(self, parents: List[Individual]) -> Individual:
        # Configs.rd is the random instance
        pos1 = Configs.rd.randint(0, len(parents) - 1)
        pos2 = Configs.rd.randint(0, len(parents) - 1)
        
        while pos1 == pos2:
            pos2 = Configs.rd.randint(0, len(parents) - 1)
            
        p1 = parents[pos1]
        p2 = parents[pos2]

        if p1.fitness > p2.fitness:
            return p1
        
        return p2

    def reproduction(self, parents: List[Individual]) -> List[Individual]:
        offspring_pop = Population(self.task)
        
        # In Python, we access the list directly, typically assuming get_population returns the list
        current_offspring = offspring_pop.get_population() 

        # The Java logic loops until the size fills up.
        # Note: In Java, if crossover doesn't happen, it loops again.
        while len(current_offspring) < Configs.POPULATION_SIZE:
            p1 = self.select_parent(parents)
            p2 = self.select_parent(parents)

            if Configs.rd.random() < Configs.CROSSOVER_RATE:
                # Assuming crossover logic is in Population class as per Java code
                o = offspring_pop.crossover(p1, p2)
                
                if Configs.rd.random() < Configs.MUTATION_RATE:
                    o = offspring_pop.mutation(o)
                
                o.update_fitness(self.task)
                current_offspring.append(o)
        
        return current_offspring