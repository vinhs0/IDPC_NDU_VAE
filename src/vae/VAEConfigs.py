import random

class Configs:
    rd = random.Random()

    MAX_VALUE = (2**31 - 1) // 3
    
    POPULATION_SIZE = 32
    MAX_GENERATIONS = 100
    REPEAT = 1
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.1

    TRANSFER_INTERVAL_GEN = 250 #chỉnh cái này để check KT
    TRANSFER_BATCH_SIZE = 32
    TRAIN_EPOCH = 5