import random

class Configs:
    # A shared Random instance
    rd = random.Random()
    
    # Java 'int' is 32-bit. To match Java's Integer.MAX_VALUE exactly:
    MAX_VALUE = (2**31 - 1) // 3
    
    POPULATION_SIZE = 100
    MAX_GENERATIONS = 500
    REPEAT = 30
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.1