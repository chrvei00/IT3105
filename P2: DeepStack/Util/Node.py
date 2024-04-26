class Node:
    def __init__(self, state: object, parent: object = None, action: str = None, player_ranges: dict = None, depth: int = 0, regret_sum: dict = None, strategy_sum: dict = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.player_ranges = player_ranges
        self.children = []
        self.depth = depth
        self.player1_value = 0
        self.player2_value = 0
        self.regret_sum = regret_sum
        self.strategy_sum = strategy_sum
    
    def add_child(self, child: object):
        self.children.append(child)
        child.depth = self.depth + 1
        child.parent = self

class State:
    def __init__(self, state_type: str, bets: dict, blind: int, player_stacks: dict, table: list, to_act: str, has_raised: dict = None, has_called: dict = None):
        self.type = state_type
        self.bets = bets
        self.blind = blind
        self.player_stacks = player_stacks
        self.table = table
        self.to_act = to_act
        self.stage = len(table)
        self.has_raised = has_raised
        self.has_called = has_called

    def __repr__(self):
        return f"State: {self.type}, Bets: {self.bets}, Player Stacks: {self.player_stacks}, Table: {self.table}, To Act: {self.to_act}, Stage: {self.stage}, Has Raised: {self.has_raised}, Has Called: {self.has_called}"