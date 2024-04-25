class Node:
    def __init__(self, state: object, parent: object = None, action: str = None, player_ranges: dict = None, depth: int = 0):
        self.state = state
        self.parent = parent
        self.action = action
        self.player_ranges = player_ranges
        self.children = []
        self.depth = depth
    
    def add_child(self, child: Node):
        self.children.append(child)
        Node.depth = self.depth + 1
        child.parent = self

class State:
    def __init__(self, state_type: str, bets: dict, blind: int, player_stacks: dict, table: list, to_act: str, has_raised: dict = None, has_called: dict = None):
        self.type = state_type
        self.pot = pot
        self.bets = bets
        self.player_stacks = player_stacks
        self.table = table
        self.to_act = to_act
        self.stage = len(table)
        self.has_raised = has_raised
        self.has_called = has_called

    def __repr__(self):
        return f"State: {self.to_act} {self.pot} {self.bets} {self.player_stacks} {self.table}"