import copy
import Util.Node as Node
import Util.Card as Card
import Util.Config as config

def gen_state(state: Node.State, object: object) -> Node.State:
    """
    Generate a child state depending on the action taken.
    """
    stacks_copy = copy.deepcopy(state.player_stacks)
    bets_copy = copy.deepcopy(state.bets)
    has_raised_copy = copy.deepcopy(state.has_raised)
    has_called_copy = copy.deepcopy(state.has_called)

    if type(object) == str:
        action = object
        # Select the only other player to act next
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
            # Find type of next state
            if all([stack == 0 for stack in stacks_copy]):
                state_type = "terminal"
            elif all([bet >= max(state.bets.values()) for bet in bets_copy.values()]) and len(state.table) < 5:
                state_type = "chance"
            else:
                state_type = "decision"
        elif action == "call":
            bets_copy[state.to_act] = max(state.bets.values())
            stacks_copy[state.to_act] -= max(state.bets.values()) - state.bets[state.to_act]
            # Find type of next state
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
            # Find type of next state
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
        if len(table_copy) < 5:
            return Node.State("decision", state.bets, state.blind, state.player_stacks, table_copy, state.to_act, state.has_raised, state.has_called)
        else:
            return Node.State("terminal", state.bets, state.blind, state.player_stacks, table_copy, state.to_act, state.has_raised, state.has_called)

def possible_actions(node: Node.Node) -> list:
    """
    Get all possible actions for the current state.
    """
    state = node.state
    actions = ["fold"]
    if state.player_stacks.get(state.to_act) > 0:
        actions.append("all-in")
    if state.player_stacks.get(state.to_act) >= max(state.bets.values()) - state.bets.get(state.to_act):
        actions.append("call")
    if state.player_stacks.get(state.to_act) >= max(state.bets.values()) - state.bets.get(state.to_act) + state.blind * 2 and state.has_raised[state.to_act] == False:
        actions.append("bet")
    return actions

def possible_cards(state: Node.State, max: int) -> list:
    """
    Get all possible cards for the current state.
    """
    deck = Card.Deck()
    deck.shuffle()
    cards = deck._deal(max)
    return [card for card in cards if card not in state.table]

def possible_hole_pairs(state: Node.State=None, max: int=52) -> list:
    """
    Get all possible hole card pairs for the current state.
    """
    hole_pairs = []
    deck = Card.Deck()
    for i in range(max):
        for j in range(i+1, max):
            hole_pairs.append([deck.cards[i], deck.cards[j]])
    return hole_pairs

def gen_hole_pair_matrix() -> dict:
    """
    Generate the regret sum for the current node.
    """
    actions = ["fold", "call", "bet", "all-in"]
    matrix = {}
    for pair in possible_hole_pairs():
        matrix[config.format_hole_pair(pair)] = {}
        for action in actions:
            matrix[config.format_hole_pair(pair)][action] = 0
    return matrix