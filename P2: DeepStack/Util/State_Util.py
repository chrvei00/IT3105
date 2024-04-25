import copy
import Util.Node as Node
import Util.Card as Card

def gen_state(state: Node.State, object: object) -> State:
    """
    Generate a child state depending on the action taken.
    """
    stacks_copy = copy.deepcopy(state.player_stacks)
    bets_copy = copy.deepcopy(state.bets)
    has_raised_copy = copy.deepcopy(state.has_raised)
    has_called_copy = copy.deepcopy(state.has_called)

    if type(object) == str:
        action = object
        next_player_to_act = state.player_stacks[state.to_act + 1] if state.to_act + 1 < len(state.player_ranges) else state.player_ranges[0]

        if action == "fold":
            stacks_copy.remove(state.to_act)
            bets_copy.remove(state.to_act)
            state_type = "terminal"
        elif action == "all-in":
            bets_copy[state.to_act] += stacks_copy[state.to_act]
            stacks_copy[state.to_act] = 0
            # Find type of next state
            if all([stack == 0 for stack in stacks_copy]):
                state_type = "terminal"
            elif all([bet >= max(state.bets.values()) for bet in bets.values()]) and len(state.table < 5):
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
            elif all([bet >= max(state.bets.values()) for bet in bets.values()]) and len(state.table < 5) and (has_called.get(next_player_to_act) == True or has_raised.get(next_player_to_act) == True):
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
            elif all([bet >= max(state.bets.values()) for bet in bets.values()]) and len(state.table < 5) and (has_called.get(next_player_to_act) == True or has_raised.get(next_player_to_act) == True):
                has_raised_copy[next_player_to_act] = False
                has_called_copy[next_player_to_act] = False
                has_raised_copy[state.to_act] = False
                has_called_copy[state.to_act] = False
                state_type = "chance"
            else:
                has_raised_copy[state.to_act] = True
                has_called_copy[next_player_to_act] = False
                state_type = "decision"

        
        return State(state_type, bets_copy, state.blind, stacks_copy, state.table, next_player_to_act, has_raised_copy, has_called_copy)
    else:
        card = object
        table_copy = copy.deepcopy(state.table)
        table_copy.append(card)
        if len(table_copy) < 5:
            return State("decision", state.bets, state.blind, state.player_stacks, table_copy, state.to_act, state.has_raised, state.has_called)
        else:
            return State("terminal", state.bets, state.blind, state.player_stacks, table_copy, state.to_act, state.has_raised, state.has_called)

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

def possible_cards(state: Node.State) -> list:
    """
    Get all possible cards for the current state.
    """
    deck = Card.Deck().shuffle().cards
    return [card for card in deck if card not in state.table]