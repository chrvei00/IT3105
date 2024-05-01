import random
import Util.Player as Player
import Util.Card as Card
import itertools
import collections
import Util.gui as gui
import Util.Config as config
import Util.State_Util as state_util
import Poker_Oracle as oracle

def validate_game(Num_Human_Players, Num_AI_Rollout_Players, Num_AI_Resolve_Players):
    """
    Validates the parameters for a game.

    Args:
        Num_Human_Players (int): Number of human players.
        Num_AI_Rollout_Players (int): Number of AI rollout players.
        Num_AI_Resolve_Players (int): Number of AI resolve players.

    Raises:
        ValueError: If the number of players is negative or less than 2.
        ValueError: If the total number of players exceeds 10.
        ValueError: If there are more than 2 players in a game with AI resolvers.
    """
    if Num_Human_Players < 0 or Num_AI_Rollout_Players < 0 or Num_AI_Resolve_Players < 0:
        raise ValueError("Number of players must be a non-negative integer")
    elif Num_Human_Players + Num_AI_Rollout_Players + Num_AI_Resolve_Players < 2:
        raise ValueError("There must be at least 2 players")
    elif Num_Human_Players + Num_AI_Rollout_Players > 6:
        raise ValueError("There can be at most 10 players")
    if Num_AI_Resolve_Players > 0 and Num_Human_Players + Num_AI_Resolve_Players + Num_AI_Rollout_Players > 2:
        raise ValueError("There can be at most 2 players in a game with AI resolvers")

def validate_hand(players: list, dealer: Player, deck: Card.Deck, blind: int):
    """
    Validates the parameters for a hand.

    Args:
        players (list): List of players.
        dealer (Player): The dealer player.
        deck (Card.Deck): The deck of cards.
        blind (int): The blind value.

    Raises:
        ValueError: If the number of players is less than 2 or exceeds 10.
        ValueError: If the dealer is not a player.
        ValueError: If the blind is negative.
        ValueError: If the deck is None.
    """
    if len(players) < 2:
        raise ValueError("There must be at least 2 players")
    elif len(players) > 10:
        raise ValueError("There can be at most 10 players")
    elif dealer not in players:
        raise ValueError("Dealer must be a player")
    elif blind < 0:
        raise ValueError("Blind must be a non-negative integer")
    elif deck is None:
        raise ValueError("Deck cannot be None")

def setup_game(Num_Human_Players, Num_AI_Rollout_Players, Num_AI_Resolve_Players: int, start_chips):
    """
    Sets up a game with the specified parameters.

    Args:
        Num_Human_Players (int): Number of human players.
        Num_AI_Rollout_Players (int): Number of AI rollout players.
        Num_AI_Resolve_Players (int): Number of AI resolve players.
        start_chips: The starting number of chips for each player.

    Returns:
        tuple: A tuple containing the players, dealer, deck, and blind value.
    """
    players = []
    for i in range(Num_Human_Players):
        players.append(Player.Player(f"H {i}" , "human", i))
    for i in range(Num_AI_Rollout_Players):
        players.append(Player.Player(f"AI_roll {i}", "AI_rollout", i + Num_Human_Players))
    for i in range(Num_AI_Resolve_Players):
        players.append(Player.Player(f"AI_res {i}", "AI_resolve", i + Num_Human_Players + Num_AI_Rollout_Players))
    for player in players:
        player.chips = start_chips
    dealer = random.choice(players)
    random.shuffle(players)
    deck = Card.Deck()
    blind = config.read_blind()

    return players, dealer, deck, blind

def game_over(players: list):
    """
    Checks if the game is over.

    Args:
        players (list): List of players.

    Returns:
        bool: True if the game is over, False otherwise.
    """
    return len(players) == 1

def hand_over(players: list, table: list):
    """
    Checks if the hand is over.

    Args:
        players (list): List of players.
        table (list): List of cards on the table.

    Returns:
        bool: True if the hand is over, False otherwise.
    """
    return len(players) == 1 or len(table) == 5

# ... (remaining functions omitted for brevity)
    wins = 0
    for i in range(config.read_simultation_size()):
        if oracle.simulate_table(deck, table, hand1, hand2):
            wins += 1
        else:
            wins -= 1
    return wins / config.read_simultation_size()

def get_utility_potrelative(hand: list, table: list, pot: float):
    return oracle.hole_card_rollout(table, hand, 1, cache=False, save=False)*pot
