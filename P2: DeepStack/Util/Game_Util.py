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

def round_over(players: list, high_bet: int):
    """
    Check if the round is over.

    Args:
        players (list): A list of player objects.
        high_bet (int): The highest bet in the current round.

    Returns:
        bool: True if the round is over, False otherwise.
    """
    if skip_player_actions(players):
        return True
    if len(players) == 1:
        return True
    for player in filter(lambda x: x.active_in_hand and not x.is_all_in, players):
        if player.current_bet != high_bet:
            return False
    return True

def skip_player_actions(players: list):
    """
    Checks if there is only one player who is not all-in.

    Args:
        players (list): A list of player objects.

    Returns:
        bool: True if there is only one player who is not all-in, False otherwise.
    """
    count_not_all_in = 0
    for player in players:
        if player.is_all_in == False:
            count_not_all_in += 1
    if count_not_all_in <= 1:
        return True

def end_action_round(players: list):
    """
    Determines if the action round has ended.

    Args:
        players (list): A list of player objects.

    Returns:
        bool: True if the action round has ended, False otherwise.
    """
    if len(players) == 1:
        return True
    if skip_player_actions(players):
        for player in players:
            if player.is_all_in == False and player.current_bet - get_high_bet(players) < 0:
                return False
        return True
    return False

def adjust_hand_params(players: list, pot: int):
    """
    Adjusts the hand parameters based on the players' bets and the pot.

    Args:
        players (list): A list of Player objects representing the players in the game.
        pot (int): The current pot size.

    Returns:
        int: The updated pot size after adjusting the hand parameters.
    """
    high_bet = get_high_bet(players)
    non_all_in_players = []
    for player in players:
        if player.is_all_in == False:
            non_all_in_players.append(player)
    if len(non_all_in_players) == 1:
        high_bet_excluded_non_all_in = 0
        for player in players:
            if player != non_all_in_players[0] and (player.current_bet > high_bet_excluded_non_all_in):
                high_bet_excluded_non_all_in = player.current_bet
        if high_bet_excluded_non_all_in < non_all_in_players[0].current_bet:
            non_all_in_players[0].chips += non_all_in_players[0].current_bet - high_bet_excluded_non_all_in
            pot -= non_all_in_players[0].current_bet - high_bet_excluded_non_all_in
            non_all_in_players[0].current_bet = high_bet_excluded_non_all_in
    return pot

def rotate(players: list, dealer: Player.Player):
    """
    Rotates the list of players so that the dealer becomes the last player in the list.

    Args:
        players (list): A list of Player objects representing the players in the game.
        dealer (Player): The Player object representing the dealer.

    Returns:
        list: The rotated list of players, with the dealer as the last player.
    """
    while players[-1] != dealer:
        players.append(players.pop(0))
    return players

def next_dealer(players: list, dealer: Player):
    """
    Returns the next dealer in the list of players.

    Parameters:
    players (list): A list of Player objects representing the players in the game.
    dealer (Player): The current dealer.

    Returns:
    Player: The next dealer in the list of players.
    """
    dealer_index = players.index(dealer)
    if dealer_index == len(players) - 1:
        return players[0]
    else:
        return players[dealer_index + 1]

def get_high_bet(players: list):
    """
    Returns the highest bet among the players.

    Args:
        players (list): A list of player objects.

    Returns:
        int: The highest bet among the players.
    """
    high_bet = 0
    for player in players:
        if player.current_bet > high_bet:
            high_bet = player.current_bet
    return high_bet

def get_winner(players: list, table: list):
    """
    Determines the winner(s) of the game based on the players' hands and the table cards.

    Args:
        players (list): A list of player objects.
        table (list): A list of cards on the table.

    Returns:
        list: A list of player objects representing the winner(s) of the game.
    """
    best_hand = None
    winners = []
    for player in players:
        hand = player.get_cards()
        if best_hand is None:
            best_hand = hand
            winners = [player]
        else:
            comp = compare_two_hands(hand, best_hand, table)
            if comp == -1:
                best_hand = hand
                winners = [player]
            elif comp == 0:
                winners.append(player)
    return winners

def best_hand_from_seven(cards):
    """
    Determines the best five-card hand from a given set of seven cards.

    Args:
        cards (list): A list of seven cards.

    Returns:
        tuple: A tuple containing the rank of the best hand and the key cards that determine the rank.
    """
    best_rank = ("High Card", [0])
    for combo in itertools.combinations(cards, 5):
        rank, key_cards = evaluate_hand(combo)
        if RANKS[rank] > RANKS[best_rank[0]] or (RANKS[rank] == RANKS[best_rank[0]] and key_cards > best_rank[1]):
            best_rank = (rank, key_cards)
    return best_rank

def compare_two_hands(hand1, hand2, board):
    """
    Compares the best hands of two players given their private cards and the board.

    Args:
        hand1 (list): A list of two private cards for player 1.
        hand2 (list): A list of two private cards for player 2.
        board (list): A list of five cards on the table.

    Returns:
        int: -1 if hand1 is better, 1 if hand2 is better, 0 if the hands are equal.
    """
    combined_hand1 = board + hand1
    combined_hand2 = board + hand2
    
    best_hand1 = best_hand_from_seven(combined_hand1)
    best_hand2 = best_hand_from_seven(combined_hand2)
    
    if RANKS[best_hand1[0]] > RANKS[best_hand2[0]]:
        return -1 
    elif RANKS[best_hand1[0]] < RANKS[best_hand2[0]]:
        return 1 
    else:
        if best_hand1[1] > best_hand2[1]:
            return -1
        elif best_hand1[1] < best_hand2[1]:
            return 1
        else:
            return 0


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
    """
    Evaluates a hand and returns its rank and the key cards for comparison.

    Parameters:
    hand (list): A list of Card objects representing the hand to be evaluated.

    Returns:
    tuple: A tuple containing the rank of the hand and the key cards for comparison.

    The rank of the hand can be one of the following:
    - "Royal Flush"
    - "Straight Flush"
    - "Four of a Kind"
    - "Full House"
    - "Flush"
    - "Straight"
    - "Three of a Kind"
    - "Two Pair"
    - "One Pair"
    - "High Card"

    The key cards for comparison are a list of integers representing the values of the cards in the hand.

    """
    values = sorted([int(card.get_real_value()) for card in hand], reverse=True)
    suits = [card.suit for card in hand]
    value_counts = collections.Counter(values)
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
    """
    Compares two hands and returns which hand is better.

    Args:
        hand1 (list): The first hand to compare.
        hand2 (list): The second hand to compare.

    Returns:
        int: -1 if hand1 is better, 1 if hand2 is better, 0 if the hands are tied.
    """
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
    """
    Visualizes the AI player's information on the game window.

    Args:
        window: The game window to display the AI player's information.
        table (list): The current state of the game table.
        name (str): The name of the AI player.
        chips (int): The number of chips the AI player has.
        pot (int): The current size of the pot.
        current_bet (int): The AI player's current bet.
        high_bet (int): The highest bet made by any player in the current round.

    Returns:
        None
    """
    gui.visualize_AI(window, table, name, chips, pot, current_bet, high_bet)

def visualize_human(window, table: list, cards: list, name: str, chips: int, pot: int, current_bet: int, high_bet: int, actions: list):
    """
    Visualizes the game state for a human player.

    Args:
        window: The game window object.
        table: A list of cards on the table.
        cards: A list of the player's hole cards.
        name: The name of the player.
        chips: The number of chips the player has.
        pot: The current size of the pot.
        current_bet: The current bet amount.
        high_bet: The highest bet amount.
        actions: A list of possible actions for the player.

    Returns:
        The visualization of the game state.

    """
    return gui.visualize_human(window, table, cards, name, chips, pot, current_bet, high_bet, actions)

def get_string_representation_cards(cards: list):
    """
    Converts a list of Card objects into a list of string representations.

    Args:
        cards (list): A list of Card objects.

    Returns:
        list: A list of string representations of the cards, where each string
              consists of the card's value followed by its suit.
    """
    return [f"{card.get_value()}{card.get_suit()}" for card in cards]

def generate_ranges():
    """
    Generates ranges using the state_util.gen_range() function.

    Returns:
        tuple: A tuple containing two ranges generated by state_util.gen_range().
    """
    return state_util.gen_range(), state_util.gen_range()

def get_utility(hand1: list, hand2: list = None, table: list = None):
    """
    Calculates the utility of a given hand in a poker game.

    Args:
        hand1 (list): The first player's hand.
        hand2 (list, optional): The second player's hand. Defaults to None.
        table (list, optional): The cards on the table. Defaults to None.

    Returns:
        float: The utility value, representing the probability of winning.
    """
    deck = Card.Deck()
    if table is None:
        table = []
    if hand2 is None:
        hand2 = []
    deck.cards = [card for card in deck.cards if card not in hand1 + hand2 + table]
    wins = 0
    for i in range(config.read_simultation_size()):
        if oracle.simulate_table(deck, table, hand1, hand2):
            wins += 1
        else:
            wins -= 1
    return wins / config.read_simultation_size()

def get_utility_potrelative(hand: list, table: list, pot: float):
    """
    Calculates the utility of a hand relative to the pot size.

    Args:
        hand (list): The player's hand.
        table (list): The cards on the table.
        pot (float): The current pot size.

    Returns:
        float: The utility of the hand relative to the pot size.
    """
    return oracle.hole_card_rollout(table, hand, 1, cache=False, save=False)*pot
