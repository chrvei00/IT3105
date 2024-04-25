import Util.resolver_util as util
import State_Manager as sm

def resolve(State: State, player1_range: dict, player2_range: dict, end_stage: str, end_depth: int, rollouts: int):
    root = sm.subtree_generator(State, end_stage, end_depth)
    for i in range(rollouts):
        # Subtree traversal rollout
        player1_value, player2_value = traverse(root, player1_range, player2_range, end_stage, end_depth)
        # Update strategy
        sigma = strategy_updater(root)
        # Generate average strategy
        sigma_mean = sigma/rollouts
        action = max(sigma_mean)
        # Update range of acting player
        player1_range = bayesian_range_updater(player1_range, action)
        # Return params
        return action, updated_state, updated_player1_range, player2_range

def travese(root, player1_range, player2_range, end_stage, end_depth):
    if util.is_showdown(state):
        return util.get_showdown_value(state)
    elif util.is_end(state, end_stage, end_depth):
        return util.get_end_stage_value(state)
    elif util.is_decision(state):
        return util.get_decision_value(state)
    else:
        for i in range(len(root.children)):
            # Traverse children
            player1_value, player2_value = traverse(root.children[i], player1_range, player2_range, end_stage, end_depth)
            # Normalize values
            player1_value = regret_updater(player1_value)
            player2_value = regret_updater(player2_value)
            # Update values
            root.children[i].player1_value = player1_value
            root.children[i].player2_value = player2_value
        return player1_value, player2_value

def update_strategy(root):
    state = root.state
    for nodes in root.children:
        update_strategy(node)
    if util.is_decision(state):
        for pair in hole_pairs:
            for action in actions:
                regret = max(0, update_regret(pair, action))
                root.regret_sum[pair][action] += regret
        for pair in hole_pairs:
            for action in actions:
                root.strategy_sum[pair][action] += root.regret_sum[pair][action]
    return root.strategy_sum

def BayesianRangeUpdater(current_range, observed_action, strategy):
    updated_range = {}
    total_probability_of_action = 0

    # Calculate the total probability of the observed action
    for hand in current_range:
        total_probability_of_action += strategy[hand][observed_action] * current_range[hand]

    # Update the range using Bayes' theorem
    for hand in current_range:
        likelihood = strategy[hand][observed_action]
        prior = current_range[hand]
        posterior = (likelihood * prior) / total_probability_of_action
        updated_range[hand] = posterior

    # Normalize the updated range to ensure it sums to 1
    normalization_factor = sum(updated_range.values())
    for hand in updated_range:
        updated_range[hand] /= normalization_factor

    return updated_range

def update_regret():
    pass

def gen_neural_net():
    pass

def train_neural_net():
    pass
