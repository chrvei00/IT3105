import random
from Util.Player import Player
from Util.Card import Deck

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
        players.append(Player(f"H{i}" , "human"))
    # Create AI players
    for i in range(Num_AI_Players):
        players.append(Player(f"AI{i}", "AI"))
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
    winner = None
    for player in players:
        hand = player.cards + table
        if best_hand is None or handIsBetter(hand, best_hand):
            best_hand = hand
            winner = player
    return winner

def handIsBetter(hand1: list, hand2: list):
    return True