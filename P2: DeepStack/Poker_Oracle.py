import copy
import Util.Player
import Util.Oracle_Util as oracle_util
import Util.Game_Util as game_util
import Util.Config as config
import Util.Card as Card

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
