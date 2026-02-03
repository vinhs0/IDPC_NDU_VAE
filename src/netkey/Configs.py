import random

# Java's Integer.MAX_VALUE is (2^31 - 1). 
# We explicitly calculate it to maintain the exact mathematical logic.
_JAVA_MAX_INT = 2**31 - 1

class Configs:
    # Random instance (equivalent to public static Random rd)
    rd = random.Random()

    # Configuration Constants
    MAX_VALUE = _JAVA_MAX_INT // 3
    POPULATION_SIZE = 100
    MAX_GENERATIONS = 500
    REPEAT = 30
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.1
    
    # New parameters
    mum = 5.0
    mu = 2.0