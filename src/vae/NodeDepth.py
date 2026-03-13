class NodeDepth:
    def __init__(self, node_or_other=0, depth=0):
        """
        Handles three Java constructors:
        1. NodeDepth()             -> NodeDepth()
        2. NodeDepth(int, int)     -> NodeDepth(1, 5)
        3. NodeDepth(NodeDepth)    -> NodeDepth(existing_obj)
        """
        if isinstance(node_or_other, NodeDepth):
            # Copy constructor logic
            self.node = node_or_other.node
            self.depth = node_or_other.depth
        else:
            # Standard constructor logic
            self.node = node_or_other
            self.depth = depth

    # Equivalent to toString()
    def __str__(self):
        return f"[{self.node}, {self.depth}]"

    # __repr__ is used when printing lists of these objects
    def __repr__(self):
        return self.__str__()