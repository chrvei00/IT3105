import copy
import numpy as np
import json
import Util.Player
import Util.Oracle_Util as oracle_util
import Util.Game_Util as game_util
import Util.Config as config
import Util.Card as Card
import Util.State_Util as state_util

def cheat_sheet() -> dict:
    """
    Return a dictionary of the optimal strategies for each state of the game.
    """
    return config.read_cheat_sheet()

def hole_card_rollout(init_table: list, hand: list, opponents: int, init_deck: object = None, cache: bool=True, save: bool=True) -> float:
    """
    Perform a hole card rollout and return the utility matrix.
    """
    # Check if rollout already performed
    if cache:
        wp = config.read_cheat_sheet().get(config.format_hand(hand))
        if wp is not None:
            return wp
    n = config.read_simultation_size()
    # Create a deck
    if init_deck is None:
        init_deck = Card.Deck()
        for card in hand + init_table:
            if card in init_deck.cards:
                init_deck.cards.remove(card)
    # Create players and opponents
    player = Util.Player.Player("Player", None)
    player.cards = hand
    opponents = [Util.Player.Player(f"Opponent {i}", None) for i in range(opponents)]
    # Simulate n times
    wins = 0;
    for _ in range(n):
        # Create a copy of the table and deck
        table = copy.deepcopy(init_table)
        deck = copy.deepcopy(init_deck)
        # Simulate the showdown
        if oracle_util.simulate(deck, table, player, opponents):
            wins += 1
    # Return the utility matrix
    if save:
        config.write_cheat_sheet(hand, opponents, wins, n)
    return wins / n

def simulate_table(init_deck, init_table: list, player_hand: list, opponent: list) -> float:
    """
    Simulate a table with a player and an opponent.
    """
    # Create a copy of the table and deck
    table = copy.deepcopy(init_table)
    deck = copy.deepcopy(init_deck)
    if init_deck is None:
        deck = Card.Deck()
        for card in player_hand + opponent + table:
            if card in deck.cards:
                deck.cards.remove(card)
    # Simulate the showdown
    deck.shuffle()
    if len(opponent) < 2:
        cards = deck.deal_card(2 - len(opponent))
        for card in cards:
            opponent.append(card)
    if len(table) < 5:
        cards = deck.deal_card(5 - len(table))
        for card in cards:
            table.append(card)
    return game_util.compare_two_hands(player_hand, opponent, table)

def generate_utility_matrix(save: bool=True, cache: bool=False):
    if cache:
        return load_utility_matrix_from_json()
    utility_matrix = {}
    hole_pairs = state_util.possible_hole_pairs()
    for i, pair1 in enumerate(hole_pairs):
        utility_matrix[oracle_util.represent_hand_as_string(pair1)] = {}
        for j, pair2 in enumerate(hole_pairs):
            if i != j:
                if any(card1.get_real_value() == card2.get_real_value() and card1.get_suit() == card2.get_suit() for card1 in pair1 for card2 in pair2):
                    utility_matrix[oracle_util.represent_hand_as_string(pair1)][oracle_util.represent_hand_as_string(pair2)] = 0
                else:
                    
                    win_probability = get_win_probability(None, [], pair1, pair2)
                    utility = 2 * (win_probability - 0.5)
                    utility_matrix[oracle_util.represent_hand_as_string(pair1)][oracle_util.represent_hand_as_string(pair2)] = utility
            else:
                utility_matrix[oracle_util.represent_hand_as_string(pair1)][oracle_util.represent_hand_as_string(pair2)] = 0
    if save:
        save_utility_matrix_to_json(utility_matrix)
    return utility_matrix

def display_utility_matrix():
    """
    Display the utility matrix.
    """
    utility_matrix = generate_utility_matrix()
    print("Utility Matrix:")
    for key1, value1 in utility_matrix.items():
        print(f"{key1}:")
        for key2, value2 in value1.items():
            print(f"    {key2}: {value2}")

def get_win_probability(deck, table, hand1, hand2):
    """
    Get the win probability of a hand against another hand.
    """
    wins = 0
    simulation_size = config.read_simultation_size()
    for i in range(simulation_size):
        wins += max(0, simulate_table(deck, table, hand1, hand2))
    return wins / simulation_size

def save_utility_matrix_to_json(utility_matrix):
    with open("utility_matrix.json", 'w') as file:
        json.dump(utility_matrix, file, indent=4)

def load_utility_matrix_from_json():
    with open("utility_matrix.json", 'r') as file:
        return json.load(file)

