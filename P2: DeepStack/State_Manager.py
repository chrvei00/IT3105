from Util.Search_Tree import State
from Util.State_Util import gen_state_call, gen_state_fold, gen_state_bet, gen_state_all_in, possible_actions

def generate_root_state(player_ranges: dict, pot: int, current_bet: int, player_stacks: dict, table: list, history: list, to_act: str) -> State:
    """
    Generate the root state for the search tree.
    """
    return State(player_ranges, pot, current_bet, player_stacks, table, history, to_act)

def generate_child_state(state: State, action: tuple) -> State:
    """
    Generate a child state depending on the action taken.
    """
    if action[0] not in possible_actions(state):
        raise ValueError("Action is not possible in the current state")

    return gen_state(state, action)

def generate_child_states(tree: Search_Tree, state: State) -> list:
    """
    Generate all child states for the search tree.
    """
    child_states = []
    for action in possible_actions(state):
        child_states.append(generate_child_state(state, action))
    return child_states