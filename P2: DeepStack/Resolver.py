import Util.resolver_util as util
import Util.State_Util as su
import Util.Game_Util as gu
import State_Manager as sm
import Util.Node as Node
import Util.Config as config
import Neural_Net as nn

class Resolver:
    def __init__(self):
        self.neural_net = nn.Neural_Net()
        self.root = None

    def get_action(self, player, state) -> str:
        player_range = player.player_range
        opponent_range = player.opponent_range
        end_stage = "terminal"
        action, player.player_range, player.opponent_range = self.resolve(state, player_range, opponent_range, end_stage, config.read_end_depth(), config.read_rollouts(), player.cards)
        if self.root:
            for child in self.root.children:
                if child.action == action:
                    self.root = child
                    update_depth(self.root)
                    if self.root.state.type == "terminal" or self.root.state.type == "chance":
                        self.root = None
                    else:
                        visualize_tree(self.root)
                        branch_tree(self.root)
                        print("Root node updated after action: ", action)
                    return action
                self.root = None
        return action

    def resolve(self, State: Node.State, player_range: dict, opponent_range: dict, end_stage: str, end_depth: int, rollouts: int, cards: list):
        if not self.root:
            "Generated inital subtree"
            self.root = sm.subtree_generator(State, end_stage, end_depth, rollouts)
        visualize_tree(self.root)
        strategy = []
        for i in range(rollouts):
            player_value, opponent_value = self.traverse(self.root, player_range, opponent_range, end_stage, end_depth)
            strategy.append(update_strategy(self.root))
        average_strategy = util.average_strategy(strategy)
        action = max(average_strategy[config.format_hole_pair(cards)], key=average_strategy[config.format_hole_pair(cards)].get)
        updated_player_range = bayesian_range_updater(player_range, action, average_strategy)
        # input(f"Action: {action}\nPress Enter to continue...")
        return action, updated_player_range, opponent_range

    def traverse(self, node, player_range, opponent_range, end_stage, end_depth):
        if node.depth == end_depth or node.state.type == end_stage:
            return self.neural_network(node.state, player_range, opponent_range)
        elif node.state.type == "decision":
            for child in node.children:
                node.player_range = bayesian_range_updater(player_range, child.action, node.strategy_sum)
                node.player_value, node.opponent_value = self.traverse(child, player_range, opponent_range, end_stage, end_depth)
        return node.player_value, node.opponent_value
    
    def neural_network(self, state, player_range, opponent_range):
        if config.read_nn_evalution():
            self.neural_net.evaluate(state, player_range, opponent_range)
        return heuristic_evaluation(state, player_range, opponent_range)

def update_strategy(node):
    state = node.state
    if state.type != "decision":
        return
    for node in node.children:
        update_strategy(node)
    for pair in node.regret_sum:
        for action in config.get_actions():
            regret_util = util.expected_payoff(pair, node.action, state.table, sum(node.state.bets.values())) - util.get_best_alternative_payoff(pair, node.state.table, sum(node.state.bets.values()))
            regret = max(0, -regret_util)
            node.regret_sum[pair][action] += regret
    for pair in node.strategy_sum:
        for action in config.get_actions():
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

def update_depth(node):
    node.depth -= 1
    if len(node.children) == 0:
        return
    for child in node.children:
        update_depth(child)

def visualize_tree(node, indent=0):
    # Print the current node's action with indentation corresponding to its depth
    print(' ' * indent * 2 + f"Depth {node.depth}, Action: {node.action}, Type: {node.state.type}, To_act: {node.state.to_act}")
    
    # If the node has children, recursively visualize each child
    if hasattr(node, 'children') and node.children:
        for child in node.children:
            visualize_tree(child, indent + 1)

def branch_tree(node, end_depth: int = config.read_end_depth(), rollouts: int = 3):
    sm.generate_children(node, end_depth, rollouts)