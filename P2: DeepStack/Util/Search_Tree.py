class Search_Tree:
    def __init__(self, root: object):
        self.root = root
        self.nodes = []
        self.nodes.append(root)
    
    def add_node(self, state: object, parent: object, action: str):
        new_node = node(state, parent, action)
        self.nodes.append(new_node)
        parent.children.append(new_node)
    
    def get_node(self, state: object) -> object:
        for node in self.nodes:
            if node.state == state:
                return node
        return None

class Node:
    def __init__(self, node_type: str, state: object, parent: object = None, action: str = None):
        self.node_type = node_type
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
    
    def add_child(self, child: object):
        self.children.append(child)
        child.parent = self

class State:
    def __init__(self, player_ranges: dict, bets: dict, blind: int, player_stacks: dict, table: list, history: list, to_act: str):
        self.pot = pot
        self.bets = bets
        self.player_stacks = player_stacks
        self.table = table
        self.history = history
        self.to_act = to_act
        self.player_ranges = player_ranges

    def __repr__(self):
        return f"State: {self.to_act} {self.pot} {self.bets} {self.player_stacks} {self.table} {self.history} {self.player_ranges}"