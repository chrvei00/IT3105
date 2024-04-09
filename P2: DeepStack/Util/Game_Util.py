import random
from Util.Player import Player
from Util.Card import Deck
from itertools import combinations
from collections import Counter
import Util.gui as gui

def validate_game(Num_Human_Players, Num_AI_Players, Game_Type):
    if Num_Human_Players < 0 or Num_AI_Players < 0:
        raise ValueError("Number of players must be a non-negative integer")
    elif Num_Human_Players + Num_AI_Players < 2:
        raise ValueError("There must be at least 2 players")
    elif Num_Human_Players + Num_AI_Players > 6:
        raise ValueError("There can be at most 10 players")
    elif Game_Type not in ["simple", "complex"]:
        raise ValueError("Game type must be either simple or complex")

def validate_hand(players: list, dealer: Player, deck: Deck, blind: int):
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

def setup_game(Num_Human_Players, Num_AI_Players, Game_Type, start_chips):
    players = []
    # Create human players
    for i in range(Num_Human_Players):
        players.append(Player(f"H {i}" , "human"))
    # Create AI players
    for i in range(Num_AI_Players):
        players.append(Player(f"AI {i}", "AI"))
    # Give each player starting chips
    for player in players:
        player.chips = start_chips
    # Randomly select a dealer
    dealer = random.choice(players)
    # Shuffle the players
    random.shuffle(players)
    # Create a deck
    deck = Deck()
    # Create a blind
    blind = 10

    return players, dealer, deck, blind

def game_over(players: list):
    return len(players) == 1

def hand_over(players: list, table: list):
    return len(players) == 1 or len(table) == 5

def round_over(players: list, high_bet: int):
    for player in players:
        if player.current_bet != high_bet:
            return False
    return True

def rotate(players: list, dealer: Player):
    while players[-1] != dealer:
        players.append(players.pop(0))
    return players

def next_dealer(players: list, dealer: Player):
    dealer_index = players.index(dealer)
    if dealer_index == len(players) - 1:
        return players[0]
    else:
        return players[dealer_index + 1]

def get_high_bet(players: list):
    high_bet = 0
    for player in players:
        if player.current_bet > high_bet:
            high_bet = player.current_bet
    return high_bet

def get_winner(players: list, table: list):
    best_hand = None
    winner = []
    for player in players:
        hand = player.get_cards()
        if best_hand is None:
            best_hand = hand
            winner = [player]
        else:
            comp = compare_two_hands(hand, best_hand, table)
            if comp == -1:
                best_hand = hand
                winner = [player]
            elif comp == 0:
                winner.append(player)
    return winner

def best_hand_from_seven(cards):
    """Given seven cards, returns the best five-card hand."""
    best_rank = ("High Card", [0])
    for combo in combinations(cards, 5):
        rank, key_cards = evaluate_hand(combo)
        if RANKS[rank] > RANKS[best_rank[0]] or (RANKS[rank] == RANKS[best_rank[0]] and key_cards > best_rank[1]):
            best_rank = (rank, key_cards)
    return best_rank

def compare_two_hands(hand1, hand2, board):
    """Compares the best hands of two players given their private cards and the board."""
    combined_hand1 = board + hand1
    combined_hand2 = board + hand2
    
    best_hand1 = best_hand_from_seven(combined_hand1)
    best_hand2 = best_hand_from_seven(combined_hand2)
    
    if RANKS[best_hand1[0]] > RANKS[best_hand2[0]]:
        return -1  # Hand1 is better
    elif RANKS[best_hand1[0]] < RANKS[best_hand2[0]]:
        return 1  # Hand2 is better
    else:  # Same rank, compare key cards
        if best_hand1[1] > best_hand2[1]:
            return -1
        elif best_hand1[1] < best_hand2[1]:
            return 1
        else:
            return 0  # Tie


RANKS = {
    "Royal Flush": 10,
    "Straight Flush": 9,
    "Four of a Kind": 8,
    "Full House": 7,
    "Flush": 6,
    "Straight": 5,
    "Three of a Kind": 4,
    "Two Pair": 3,
    "One Pair": 2,
    "High Card": 1
}

def evaluate_hand(hand):
    """Evaluates a hand and returns its rank and the key cards for comparison."""
    values = sorted([card.get_real_value() for card in hand], reverse=True)
    suits = [card.suit for card in hand]
    value_counts = Counter(values)
    is_flush = len(set(suits)) == 1
    is_straight = all([values[i] - values[i+1] == 1 for i in range(len(values)-1)]) or values == [14, 5, 4, 3, 2, 1]

    if is_flush and is_straight and values[0] == 14:
        return ("Royal Flush", values)
    elif is_flush and is_straight:
        return ("Straight Flush", values)
    elif 4 in value_counts.values():
        four = [v for v, count in value_counts.items() if count == 4]
        return ("Four of a Kind", four + [max(v for v, count in value_counts.items() if count < 4)])
    elif sorted(value_counts.values()) == [2, 3]:
        three = [v for v, count in value_counts.items() if count == 3]
        two = [v for v, count in value_counts.items() if count == 2]
        return ("Full House", three + two)
    elif is_flush:
        return ("Flush", values)
    elif is_straight:
        return ("Straight", values)
    elif 3 in value_counts.values():
        three = [v for v, count in value_counts.items() if count == 3]
        return ("Three of a Kind", three + [v for v in values if v not in three])
    elif sorted(value_counts.values())[-2:] == [2, 2]:
        pairs = sorted([v for v, count in value_counts.items() if count == 2], reverse=True)
        return ("Two Pair", pairs + [max(v for v in values if v not in pairs)])
    elif 2 in value_counts.values():
        pair = [v for v, count in value_counts.items() if count == 2]
        return ("One Pair", pair + [v for v in values if v not in pair])
    else:
        return ("High Card", values)

def compare_hands(hand1, hand2):
    """Compares two hands and returns which hand is better (-1 for hand1, 1 for hand2, 0 for tie)."""
    rank1, key_cards1 = evaluate_hand(hand1)
    rank2, key_cards2 = evaluate_hand(hand2)
    if RANKS[rank1] > RANKS[rank2]:
        return -1
    elif RANKS[rank1] < RANKS[rank2]:
        return 1
    else:  # Ranks are equal, compare key cards
        for i in range(len(key_cards1)):
            if key_cards1[i] > key_cards2[i]:
                return -1
            elif key_cards1[i] < key_cards2[i]:
                return 1
        return 0  # Hands are completely equal

def visualize_AI(window, table: list, name: str, chips: int, pot: int, current_bet: int, high_bet: int):
    gui.visualize_AI(window, table, name, chips, pot, current_bet, high_bet)
    print(f"\n{name} has {chips} chips, the pot is {pot}, the current bet is {current_bet}, and the high bet is {high_bet}")
    print(f"\nTable: {table}\n")

def visualize_human(window, table: list, cards: list, name: str, chips: int, pot: int, current_bet: int, high_bet: int):
    gui.visualize_human(window, table, cards, name, chips, pot, current_bet, high_bet)
    input(f"\n{name}, press enter to choose your action")
    print(f"\n{name} has {chips} chips, the pot is {pot}, the current bet is {current_bet}, and the high bet is {high_bet}")
    print(f"To call: {high_bet - current_bet}")
    print(f"\nTable: {table}")
    print(f"Your cards: {cards}\n")

def get_string_representation_cards(cards: list):
    return [f"{card.get_value()}{card.get_suit()}" for card in cards]