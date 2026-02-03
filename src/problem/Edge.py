class Edge:
    def __init__(self, node1=0, node2=0):
        # This handles both Edge() and Edge(int, int)
        self._node1 = node1
        self._node2 = node2

    def get_node1(self):
        return self._node1

    def set_node1(self, node1):
        self._node1 = node1

    def get_node2(self):
        return self._node2

    def set_node2(self, node2):
        self._node2 = node2