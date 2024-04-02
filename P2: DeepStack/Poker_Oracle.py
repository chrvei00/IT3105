import copy
from Util.Oracle_Util import simulate
from Util.Player import Player
from Util.Config import read_simultation_size, read_cheat_sheet, write_cheat_sheet, format_hand


def cheat_sheet() -> dict:
    """
    Return a dictionary of the optimal strategies for each state of the game.
    """
    return read_cheat_sheet()

def hole_card_rollout(init_table: list, hand: list, opponents: int, init_deck: object) -> float:
    """
    Perform a hole card rollout and return the utility matrix.
    """
    # Check if rollout already performed
    wp = read_cheat_sheet().get(format_hand(hand))
    if wp is not None:
        return wp
    n = read_simultation_size()
    # Create players and opponents
    player = Player("Player", None)
    player.cards = hand
    opponents = [Player(f"Opponent {i}", None) for i in range(opponents)]
    # Simulate n times
    wins = 0;
    for _ in range(n):
        # Create a copy of the table and deck
        table = copy.deepcopy(init_table)
        deck = copy.deepcopy(init_deck)
        # Simulate the showdown
        if simulate(deck, init_table, player, opponents):
            wins += 1
    # Return the utility matrix
    write_cheat_sheet(hand, wins, n)
    return wins / n
