import Util.resolver_util as util
import Util.State_Util as su
import State_Manager as sm
import Util.Node as Node
import Util.Config as config

def get_action(player, state) -> str:
    player1_range = player.player1_range
    player2_range = player.player2_range
    end_stage = "terminal"
    end_depth = config.read_end_depth()
    rollouts = config.read_rollouts()
    action, updated_player1_range, player2_range = resolve(state, player1_range, player2_range, end_stage, end_depth, rollouts)
    player.player1_range = updated_player1_range
    player.player2_range = player2_range
    return action



def resolve(State: Node.State, player1_range: dict, player2_range: dict, end_stage: str, end_depth: int, rollouts: int):
    root = sm.subtree_generator(State, end_stage, end_depth, rollouts)
    for i in range(rollouts):
        # Subtree traversal rollout
        player1_value, player2_value = traverse(root, player1_range, player2_range, end_stage, end_depth)
        # Update strategy
        strategy = update_strategy(root)
        # Generate average strategy
        # TODO: Vi har kommet hit men ikke lenger. Jeg må finne ut hvordan man henter ut en action fra en strategi
        action = max(strategy.values(), key=lambda x: x["fold"])
        #TODO: We need to get only one action from the strategy
        print(action)
        # Update range of acting player
        updated_player1_range = bayesian_range_updater(player1_range, action, strategy)
        # Return params
    return action, updated_player1_range, player2_range

def traverse(node, player1_range, player2_range, end_stage, end_depth):
    if node.depth == end_depth and node.state.type == end_stage:
        return neural_network(node.state, player1_range, player2_range)
    elif node.state.type == end_stage:
        # TODO match range with table
        return 0, 0
    else:
        for i in range(len(node.children)):
            # Traverse children
            player1_value, player2_value = traverse(node.children[i], player1_range, player2_range, end_stage, end_depth)
            # Update values
            node.children[i].player1_value = player1_value
            node.children[i].player2_value = player2_value
        return node.player1_value, node.player2_value

def update_strategy(node):
    state = node.state
    for node in node.children:
        update_strategy(node)
    if state.type == "decision":
        for pair in su.possible_hole_pairs(state.table):
            tup_pair = config.tuple_hole_pair(pair)
            for action in su.possible_actions(node):
                try:
                    regret = max(0, node.regret_sum[tup_pair][action] + 0) #TODO add utility of action - node.player1_value)
                    node.regret_sum[pair][action] += regret
                except KeyError:
                    pass
                    # print("KeyError: ", pair, action)
        for pair in su.possible_hole_pairs(state.table):
            tup_pair = config.tuple_hole_pair(pair)
            for action in su.possible_actions(node):
                try:
                    node.strategy_sum[tup_pair][action] += node.regret_sum[tuple(pair)][action]
                except KeyError:
                    pass
                    # print("KeyError: ", pair, action)
    return node.strategy_sum

def bayesian_range_updater(current_range, observed_action, strategy):
    updated_range = {}
    total_probability_of_action = 0

    # Calculate the total probability of the observed action
    for hand in current_range:
        try:
            total_probability_of_action += strategy[hand][observed_action] * current_range[hand]
        except KeyError:
            pass

    # Update the range using Bayes' theorem
    for hand in current_range:
        try:
            likelihood = strategy[hand][observed_action]
            prior = current_range[hand]
            posterior = (likelihood * prior) / total_probability_of_action
            updated_range[hand] = posterior
        except KeyError:
            pass
    # Normalize the updated range to ensure it sums to 1
    normalization_factor = sum(updated_range.values())
    for hand in updated_range:
        updated_range[hand] /= normalization_factor

    return updated_range

def neural_network(state, player1_range, player2_range):
    return 0, 0