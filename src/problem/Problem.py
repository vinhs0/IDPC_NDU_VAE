from typing import List
from .IDPCNDU import IDPCNDU

class Problem:
    def __init__(self):
        self.tasks: List[IDPCNDU] = []
        self.TASKS_NUM: int = 0
        self.number_of_domains: int = 0
        self.adj_domain: List[List[int]] = []
        self.parent_domain: List[List[int]] = []
        self.number_of_domain_edge: int = 0

    def get_task(self, task_id: int) -> IDPCNDU:
        return self.tasks[task_id]

    def add_task(self, task: IDPCNDU):
        # Update max edges found so far
        self.number_of_domain_edge = max(len(task.edges_domain), self.number_of_domain_edge)
        
        self.tasks.append(task)
        self.TASKS_NUM += 1

        task_domains_count = task.get_number_of_domains()

        if self.number_of_domains < task_domains_count:
            self.number_of_domains = task_domains_count
            
            # Merge existing domains
            for i in range(len(self.adj_domain)):
                self._merge_lists(self.adj_domain[i], task.adj_domain[i])
                self._merge_lists(self.parent_domain[i], task.parent_domain[i])
            
            # Add new domains from the task
            # Java: for(int i = adjDomain.size(); i <= numberOfDomains; i++)
            # This covers indices from old_size to new_max_size (inclusive)
            for i in range(len(self.adj_domain), self.number_of_domains + 1):
                # Create deep copies of the lists
                self.adj_domain.append(list(task.adj_domain[i]))
                self.parent_domain.append(list(task.parent_domain[i]))
                
        else:
            # Task has fewer or equal domains than current problem state
            # Java: for(int i = 0; i < task.getNumberOfDomains(); i++)
            # Note: This strictly follows the Java loop (0 to N-1). 
            # If the domain system is 1-based, index N might be skipped here as per original code.
            for i in range(task_domains_count):
                self._merge_lists(self.adj_domain[i], task.adj_domain[i])
                self._merge_lists(self.parent_domain[i], task.parent_domain[i])

        # Assert size is not equal to number_of_domains (Since size is usually N+1)
        assert len(self.adj_domain) != self.number_of_domains

    def _merge_lists(self, target_list: List[int], source_list: List[int]):
        """
        Helper to replicate Java's:
        temp = new ArrayList<>(source);
        temp.removeAll(target);
        target.addAll(temp);
        
        Effect: Appends elements from source to target if they are not already in target.
        """
        for item in source_list:
            if item not in target_list:
                target_list.append(item)

    # Getters and Setters
    def get_tasks(self) -> List[IDPCNDU]:
        return self.tasks

    def set_tasks(self, tasks: List[IDPCNDU]):
        self.tasks = tasks

    def get_tasks_num(self) -> int:
        return self.TASKS_NUM

    def set_tasks_num(self, tasks_num: int):
        self.TASKS_NUM = tasks_num

    def get_number_of_domains(self) -> int:
        return self.number_of_domains

    def set_number_of_domains(self, number_of_domains: int):
        self.number_of_domains = number_of_domains

    def get_adj_domain(self) -> List[List[int]]:
        return self.adj_domain

    def set_adj_domain(self, adj_domain: List[List[int]]):
        self.adj_domain = adj_domain

    def get_parent_domain(self) -> List[List[int]]:
        return self.parent_domain

    def set_parent_domain(self, parent_domain: List[List[int]]):
        self.parent_domain = parent_domain