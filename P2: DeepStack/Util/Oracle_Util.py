from Util.Game_Util import get_winner

def simulate(deck: object, table: list, player: object, opponents: list) -> bool:
    """
    Simulate a round of poker and return the winner of the hand.
    """
    deck.shuffle()
    if len(table) < 5:
        table.append(5 - deck.deal_card(len(table)))
    # Deal cards to opponents
    for opponent in opponents:
        opponent.deal_card(deck.deal_card(2))
    # Determine the winner of the hand
    winners = get_winner([player] + opponents, table)
    # Reset opponents hands
    for opponent in opponents:
        opponent.cards = []
    # Return the winner == player
    return winners.__contains__(player)