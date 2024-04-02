import copy
import numpy as np
from configparser import ConfigParser
from Util.Oracle_Util import simulate
from Util.Player import Player
from Util.Card import Deck

def evaluate_state(table: list, hand: list) -> np.matrix:
    """
    Evaluate the state of the game and return a utility matrix.
    Use rollouts to compute the utilities of the game.
    """
    pass

def cheat_sheet() -> dict:
    """
    Return a dictionary of the optimal strategies for each state of the game.
    """
    config = ConfigParser()
    config.read('config.ini')
    if not config.has_section('cheat_sheet'):
        return None
    return dict(config['cheat_sheet'])

def hole_card_rollout(init_table: list, hand: list, opponents: int, init_deck: object) -> np.matrix:
    """
    Perform a hole card rollout and return the utility matrix.
    """
    n = 10000
    # Create players and opponents
    player = Player("Player", None)
    player.cards = hand
    opponents = [Player(f"Opponent {i}", None) for i in range(opponents)]
    # Simulate n times
    wins = 0;
    for _ in range(n):
        # Create a copy of the table
        table = copy.deepcopy(init_table)
        # Create a copy of the deck
        deck = copy.deepcopy(init_deck)
        # Shuffle the deck
        deck.shuffle()
        # Deal the remaining cards to the opponents
        for opponent in opponents:
            opponent.deal_card(deck.deal_card())
            opponent.deal_card(deck.deal_card())
        # Simulate the showdown
        if simulate(deck, init_table, player, opponents):
            wins += 1
        # Reset the opponents' hands
        for opponent in opponents:
            opponent.cards = []
    # Return the utility matrix
    config = ConfigParser()
    config.read('config.ini')
    if not config.has_section('cheat_sheet'):
        config.add_section('cheat_sheet')
    hand.sort(key=lambda x: x.get_real_value(), reverse=True)
    format_hand = f"{hand[0].get_value()}{hand[1].get_value()}"
    if (hand[0].get_suit() == hand[1].get_suit()):
        suited = "Suited"
    else:
        suited = "Offsuit"
    config.set("cheat_sheet", f"{format_hand}-{suited}", f"{wins / n}")
    with open('config.ini', 'w') as configfile:
        config.write(configfile)
    return wins / n
