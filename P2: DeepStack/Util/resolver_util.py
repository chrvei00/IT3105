import Util.Card as Card
import Util.Config as config
import Util.State_Util as state_util
import Util.Game_Util as game_util
import re

def predict_value(hole_cards, table: list):
    """
    Predicts the value of a particular hand given the current table.
    
    Args:
        hole_cards (str): The hole cards as a string.
        table (list): The current table as a list of cards.
    
    Returns:
        float: The predicted value of the hand.
    """
    return game_util.get_utility(turn_hand_string_to_list(hole_cards), table)
    

def expected_payoff(hole_cards, action, table, pot):
    """
    Estimates the expected utility or payoff of taking a certain action with a certain pair of hole cards.
    
    Args:
        hole_cards (str): The hole cards as a string.
        action (str): The action to take.
        table (list): The current table as a list of cards.
        pot (float): The current pot size.
    
    Returns:
        float: The estimated expected utility or payoff.
    """
    if action == "fold":
        return -1 * pot
    return game_util.get_utility(turn_hand_string_to_list(hole_cards), table)*pot

def get_best_alternative_payoff(hole_cards, table, pot):
    """
    Computes the best alternative payoff if the best action was taken, compared to the current action.
    
    Args:
        hole_cards (str): The hole cards as a string.
        table (list): The current table as a list of cards.
        pot (float): The current pot size.
    
    Returns:
        float: The best alternative payoff.
    """
    return pot

def turn_hand_string_to_list(hand: str) -> list:
    """
    Convert a hand string to a list of cards.
    
    Args:
        hand (str): The hand as a string.
    
    Returns:
        list: The hand as a list of Card objects.
    """
    cards = re.findall(r'[♠♣♦♥](?:10|[1-9JQKA])', hand)
    card1 = Card.Card(suit=represent_suit(cards[0][0]), value=represent_value(cards[0][1:]))
    card2 = Card.Card(suit=represent_suit(cards[1][0]), value=represent_value(cards[1][1:]))
    return [card1, card2]

def represent_suit(suit: str):
    """
    Represents the suit of a card as a string.
    
    Args:
        suit (str): The suit of the card.
    
    Returns:
        str: The suit represented as a string.
    """
    if suit == "♥":
        return "Hearts"
    elif suit == "♦":
        return "Diamonds"
    elif suit == "♣":
        return "Clubs"
    elif suit == "♠":
        return "Spades"

def represent_value(value: str):
    """
    Represents the value of a card as an integer.
    
    Args:
        value (str): The value of the card.
    
    Returns:
        int: The value represented as an integer.
    """
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
    
    Args:
        strategy (list): A list of strategies.
    
    Returns:
        dict: The average strategy as a dictionary.
    """
    actions = config.get_actions()
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