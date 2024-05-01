import copy
import Util.Node as Node
import Util.Card as Card
import Util.Config as config

def gen_state(state: Node.State, object: object) -> Node.State:
    """
    Generate a child state depending on the action taken.

    Args:
        state (Node.State): The current state.
        object (object): The action or card object.

    Returns:
        Node.State: The generated child state.
    """
    stacks_copy = copy.deepcopy(state.player_stacks)
    bets_copy = copy.deepcopy(state.bets)
    has_raised_copy = copy.deepcopy(state.has_raised)
    has_called_copy = copy.deepcopy(state.has_called)

    if type(object) == str:
        action = object
        for key in state.player_stacks.keys():
            if key != state.to_act:
                next_player_to_act = key
        if action == "fold":
            del stacks_copy[state.to_act]
            del bets_copy[state.to_act]
            state_type = "terminal"
        elif action == "all-in":
            bets_copy[state.to_act] += stacks_copy[state.to_act]
            stacks_copy[state.to_act] = 0
            if all([stack == 0 for stack in stacks_copy]):
                state_type = "terminal"
            elif all([bet >= max(state.bets.values()) for bet in bets_copy.values()]) and len(state.table) < 5:
                state_type = "chance"
            else:
                state_type = "decision"
        elif action == "call":
            bets_copy[state.to_act] = max(state.bets.values())
            stacks_copy[state.to_act] -= max(state.bets.values()) - state.bets[state.to_act]
            if all([stack == 0 for stack in stacks_copy]):
                if len(state.table) < 5:
                    state_type = "chance"
                else:
                    state_type = "terminal"
            elif all([bet >= max(state.bets.values()) for bet in bets_copy.values()]) and len(state.table) < 5 and (state.has_called.get(next_player_to_act) == True or state.has_raised.get(next_player_to_act) == True):
                has_raised_copy[next_player_to_act] = False
                has_called_copy[next_player_to_act] = False
                has_raised_copy[state.to_act] = False
                has_called_copy[state.to_act] = False
                state_type = "chance"
            else:
                has_called_copy[state.to_act] = True
                state_type = "decision"
        elif action == "bet":
            bets_copy[state.to_act] = max(state.bets.values()) + state.blind * 2
            stacks_copy[state.to_act] -= max(state.bets.values()) - state.bets[state.to_act] + state.blind * 2
            has_raised_copy[state.to_act] = True
            if all([stack == 0 for stack in stacks_copy]):
                if len(state.table) < 5:
                    state_type = "chance"
                else:
                    state_type = "terminal"
            elif all([bet >= max(state.bets.values()) for bet in bets_copy.values()]) and len(state.table) < 5 and (state.has_called.get(next_player_to_act) == True or state.has_raised.get(next_player_to_act) == True):
                has_raised_copy[next_player_to_act] = False
                has_called_copy[next_player_to_act] = False
                has_raised_copy[state.to_act] = False
                has_called_copy[state.to_act] = False
                state_type = "chance"
            else:
                has_raised_copy[state.to_act] = True
                has_called_copy[next_player_to_act] = False
                state_type = "decision"

        
        return Node.State(state_type, bets_copy, state.blind, stacks_copy, state.table, next_player_to_act, has_raised_copy, has_called_copy)
    else:
        card = object
        table_copy = copy.deepcopy(state.table)
        table_copy.append(card)
        if len(table_copy) == 3 or len(table_copy) == 4 or len(table_copy) == 5:
            state_type = "decision"
        else:
            state_type = "chance"
        return Node.State(state_type, state.bets, state.blind, state.player_stacks, table_copy, state.to_act, state.has_raised, state.has_called)

def possible_actions(node: Node.Node) -> list:
    """
    Get all possible actions for the current state.

    Args:
        node (Node.Node): The current node.

    Returns:
        list: A list of possible actions.
    """
    allowed_actions = config.get_actions()
    state = node.state
    actions = ["fold"]
    if state.player_stacks.get(state.to_act) > 0 and "all-in" in allowed_actions:
        actions.append("all-in")
    if state.player_stacks.get(state.to_act) >= max(state.bets.values()) - state.bets.get(state.to_act) and "call" in allowed_actions:
        actions.append("call")
    if state.player_stacks.get(state.to_act) >= max(state.bets.values()) - state.bets.get(state.to_act) + state.blind * 2 and state.has_raised[state.to_act] == False and "bet" in allowed_actions:
        actions.append("bet")
    return actions

def possible_cards(state: Node.State, max: int = config.read_chance_cards()) -> list:
    """
    Get all possible cards for the current state.

    Args:
        state (Node.State): The current state.
        max (int): The maximum number of cards to get.

    Returns:
        list: A list of possible cards.
    """
    deck = Card.Deck()
    deck.shuffle()
    cards = deck._deal(max)
    possible_cards = []
    for card in cards:
        if not any([card.get_real_value() == table_card.get_real_value() and card.get_suit() == table_card.get_suit() for table_card in state.table]):
            possible_cards.append(card)
    return possible_cards

def possible_hole_pairs(state: Node.State=None, max: int=52) -> list:
    """
    Get all possible hole card pairs for the current state.

    Args:
        state (Node.State, optional): The current state. Defaults to None.
        max (int, optional): The maximum number of hole card pairs to get. Defaults to 52.

    Returns:
        list: A list of possible hole card pairs.
    """
    hole_pairs = []
    for card1 in Card.Card.get_all_cards():
        for card2 in Card.Card.get_all_cards():
            if not (card1.get_real_value() == card2.get_real_value() and card1.get_suit() == card2.get_suit()):
                hole_pair_sorted = sorted([card1, card2], key=lambda card: card.get_real_value(), reverse=True)
                hole_pairs.append(hole_pair_sorted)
    return hole_pairs

def gen_hole_pair_matrix(init_value: float = 0) -> dict:
    """
    Generate the regret sum for the current node.

    Args:
        init_value (float, optional): The initial value for the regret sum. Defaults to 0.

    Returns:
        dict: The generated regret sum matrix.
    """
    actions = config.get_actions()
    matrix = {}
    for pair in possible_hole_pairs():
        matrix[config.format_hole_pair(pair, sort=False)] = {}
        for action in actions:
            matrix[config.format_hole_pair(pair, sort=False)][action] = init_value
    return matrix

def gen_range() -> dict:
    """
    Generate the range matrix.

    Returns:
        dict: The generated range matrix.
    """
    matrix = {}
    num_hole_pairs = len(possible_hole_pairs())
    for pair in possible_hole_pairs():
        matrix[config.format_hole_pair(pair, sort=False)] = 1/num_hole_pairs
    return matrix