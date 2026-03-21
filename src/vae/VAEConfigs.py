import random

class Configs:
    rd = random.Random()

    MAX_VALUE = (2**31 - 1) // 3
    
    POPULATION_SIZE = 100
    MAX_GENERATIONS = 500
    REPEAT = 5
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.1

    # Parameters for batch size and after how many iterations the algorithm performs KT
    TRANSFER_INTERVAL_GEN = 10
    TRANSFER_BATCH_SIZE = 50