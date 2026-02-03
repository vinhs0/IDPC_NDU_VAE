# Assuming these imports based on your previous file structure
from .Problem import Problem
from .IDPCNDU import IDPCNDU

class ProblemConstructor:
    
    @staticmethod
    def get_pair_instances(x: str, y: str) -> Problem:
        prob = Problem()
        
        # Load first instance
        task = IDPCNDU()
        task.read_data(x)
        prob.add_task(task)
        
        # Load second instance
        task = IDPCNDU()
        task.read_data(y)
        prob.add_task(task)
        
        return prob