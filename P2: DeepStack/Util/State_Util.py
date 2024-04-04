import copy
from Util.Search_Tree import State

def gen_state(state: State, action: tuple) -> State:
    """
    Generate a child state depending on the action taken.
    """
    ranges_copy = copy.deepcopy(state.player_ranges)
    bets_copy = copy.deepcopy(state.bets)
    stacks_copy = copy.deepcopy(state.player_stacks)
    history_copy = copy.deepcopy(state.history)

    next_player_to_act = state.player_ranges[state.to_act + 1] if state.to_act + 1 < len(state.player_ranges) else state.player_ranges[0]
    new_pot = state.pot + action[1]

    if action[0] == "fold":
        ranges_copy.remove(state.to_act)
        stacks_copy.remove(state.to_act)
        bets_copy.remove(state.to_act)
    else:
        stacks_copy.update(stacks_copy.get(state.to_act) - action[1])
        bets_copy.update(bets_copy.get(state.to_act) + action[1])
    new_history = history_copy.append(f"{state.to_act} {action[0]} {action[1]}")
    return State(ranges_copy, bets_copy, state.blind, stacks_copy, state.table, history_copy, next_player_to_act)
    
def possible_actions(state: State) -> list:
    """
    Get all possible actions for the current state.
    """
    actions = [("fold", 0)]
    # Check if acting player can call
    if state.bets.get(state.to_act) < max(state.bets.values()):
        actions.append(("call", max(state.bets.values()) - state.bets.get(state.to_act)))
    # Check if acting player can bet
    if state.player_stacks.get(state.to_act) > state.bets.get(state.to_act):
        actions.append(("bet", state.blind * 2))
    # Check if acting player can go all in
    if state.player_stacks.get(state.to_act) < state.bets.get(state.to_act):
        actions.append(("all in", state.player_stacks.get(state.to_act)))
    return actions