class Node:
    """
    Represents a node in a tree structure.

    Attributes:
        state (object): The state associated with the node.
        parent (object): The parent node of the current node.
        action (str): The action taken to reach the current node.
        player_range (dict): The range of the player at the current node.
        opponent_range (dict): The range of the opponent at the current node.
        depth (int): The depth of the node in the tree.
        regret_sum (dict): The sum of regrets for each action at the current node.
        strategy_sum (dict): The sum of strategies for each action at the current node.
        player_value (dict): The value of the player at the current node.
        opponent_value (dict): The value of the opponent at the current node.
        children (list): The list of child nodes of the current node.
    """

    def __init__(self, state: object, parent: object = None, action: str = None, player_range: dict = None, opponent_range: dict = None, depth: int = 0, regret_sum: dict = None, strategy_sum: dict = None, player_value: dict = None, opponent_value: dict = None):
        self.state = state
        self.parent = parent
        self.player_range = player_range
        self.opponent_range = opponent_range
        self.children = []
        self.depth = depth
        self.action = action
        self.regret_sum = regret_sum
        self.strategy_sum = strategy_sum
        self.player_value = player_value
        self.opponent_value = opponent_value
    
    def add_child(self, child: object):
        """
        Adds a child node to the current node.

        Args:
            child (object): The child node to be added.
        """
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