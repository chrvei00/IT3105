import Util.Card as Card
import Util.Config as config
import Util.State_Util as state_util
import Util.Game_Util as game_util
import re

def predict_value(hole_cards, table: list):
    """
    Predicts the value of a particular hand given the current table.
    This simplified version uses the utility matrix to determine the hand's value.
    """
    # Assume `utility_matrix` is keyed by player 1's hands and values are dictionaries keyed by player 2's hands.
    # This value is a simplified estimate assuming equal probability of player 2's hands.
    return game_util.get_utility(turn_hand_string_to_list(hole_cards), table)
    

def expected_payoff(hole_cards, action, table, pot):
    """
    Estimates the expected utility or payoff of taking a certain action with a certain pair of hole cards.
    """
    # The expected payoff of a hand can be computed as the average utility of that hand against all possible opponent hands.
    # Here, we assume all actions have the same effect on utility, which may not be the case in a real poker game.
    if action == "fold":
        return -1 * pot
    return game_util.get_utility(turn_hand_string_to_list(hole_cards), table)*pot

def get_best_alternative_payoff(hole_cards, table, pot):
    """
    Computes the best alternative payoff if the best action was taken, compared to the current action.
    """
    # This is a simplification since in a real game you would consider different actions leading to different states.
    # Here we are just getting the best possible utility without considering actions.
    return pot

def turn_hand_string_to_list(hand: str) -> list:
    """
    Convert a hand string to a list of cards.
    """
    cards = re.findall(r'[♠♣♦♥](?:10|[1-9JQKA])', hand)
    card1 = Card.Card(suit=represent_suit(cards[0][0]), value=represent_value(cards[0][1:]))
    card2 = Card.Card(suit=represent_suit(cards[1][0]), value=represent_value(cards[1][1:]))
    return [card1, card2]

def represent_suit(suit: str):
    if suit == "♥":
        return "Hearts"
    elif suit == "♦":
        return "Diamonds"
    elif suit == "♣":
        return "Clubs"
    elif suit == "♠":
        return "Spades"

def represent_value(value: str):
    if value == "J":
        return 11
    elif value == "Q":
        return 12
    elif value == "K":
        return 13
    elif value == "A":
        return 14
    else:
        return int(value)

def average_strategy(strategy: list) -> dict:
    """
    Compute the average strategy from a list of strategies.
    """
    actions = ["fold", "call", "bet", "all-in"]
    average = state_util.gen_hole_pair_matrix()
    for strat in strategy:
        for pair in strat:
            for action in actions:
                try:
                    average[pair][action] += strat[pair][action]
                except KeyError:
                    print (f"KeyError: pair: {pair}, action: {action}")
    for pair in average:
        for action in actions:
            try:
                average[pair][action] /= len(strategy)
            except KeyError:
                print (f"KeyError: pair: {pair}, action: {action}")
    return average