import Util.resolver_util as util
import Util.State_Util as su
import Util.Game_Util as gu
import State_Manager as sm
import Util.Node as Node
import Util.Config as config

def get_action(player, state) -> str:
    player_range = player.player_range
    opponent_range = player.opponent_range
    end_stage = "terminal"
    action, player.player_range, player.opponent_range = resolve(state, player_range, opponent_range, end_stage, config.read_end_depth(), config.read_rollouts(), player.cards)
    return action


def resolve(State: Node.State, player_range: dict, opponent_range: dict, end_stage: str, end_depth: int, rollouts: int, cards: list):
    root = sm.subtree_generator(State, end_stage, end_depth, rollouts)
    strategy = []
    for i in range(rollouts):
        player_value, opponent_value = traverse(root, player_range, opponent_range, end_stage, end_depth)
        strategy.append(update_strategy(root))
    average_strategy = util.average_strategy(strategy)
    action = max(average_strategy[config.format_hole_pair(cards)], key=average_strategy[config.format_hole_pair(cards)].get)
    updated_player_range = bayesian_range_updater(player_range, action, average_strategy)
    # input(f"Action: {action}\nPress Enter to continue...")
    return action, updated_player_range, opponent_range

def traverse(node, player_range, opponent_range, end_stage, end_depth):
    if node.depth == end_depth or node.state.type == end_stage:
        return neural_network(node.state, player_range, opponent_range)
    elif node.state.type == "decision":
        for child in node.children:
            node.player_range = bayesian_range_updater(player_range, child.action, node.strategy_sum)
            node.player_value, node.opponent_value = traverse(child, player_range, opponent_range, end_stage, end_depth)
    return node.player_value, node.opponent_value

def update_strategy(node):
    state = node.state
    if state.type != "decision":
        return
    for node in node.children:
        update_strategy(node)
    actions = ["fold", "call", "bet", "all-in"]
    for pair in node.regret_sum:
        for action in actions:
            regret_util = util.expected_payoff(pair, node.action, state.table, sum(node.state.bets.values())) - util.get_best_alternative_payoff(pair, node.state.table, sum(node.state.bets.values()))
            print(f"Regret Utility: {-regret_util}")
            regret = max(0, -regret_util)
            node.regret_sum[pair][action] += regret
    for pair in node.strategy_sum:
        for action in actions:
                node.strategy_sum[pair][action] += node.regret_sum[pair][action]
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
            if total_probability_of_action == 0:
                posterior = 0
            else:
                posterior = (likelihood * prior) / total_probability_of_action
            updated_range[hand] = posterior
        except KeyError:
            pass
    # Normalize the updated range to ensure it sums to 1
    normalization_factor = sum(updated_range.values())
    for hand in updated_range:
        if normalization_factor == 0:
            updated_range[hand] = 0
        else:
            updated_range[hand] /= normalization_factor

    return updated_range

def neural_network(state, player_range, opponent_range):
    # TODO: implement neural network
    return heuristic_evaluation(state, player_range, opponent_range)

def heuristic_evaluation(state, player_range, opponent_range) -> dict:
    # Initialize valuevetors
    player_value = su.gen_range()
    opponent_value = su.gen_range()
    # Calculate the value of the player's hand
    for hand in player_range:
        player_value[hand] = gu.get_utility(hand1=util.turn_hand_string_to_list(hand), table=state.table)
    # Calculate the value of the opponent's hand
    for hand in opponent_range:
        opponent_value[hand] = -gu.get_utility(hand1=util.turn_hand_string_to_list(hand), table=state.table)
    
    return player_value, opponent_value
