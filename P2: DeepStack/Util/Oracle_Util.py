from Util.Game_Util import get_winner

def simulate(deck: object, table: list, player: object, opponents: list) -> bool:
    """
    Simulate a round of poker and return the winner of the hand.
    """
    while len(table) < 5:
        table.append(deck.deal_card())
    # Determine the winner of the hand
    players = [player]
    for opponent in opponents:
        players.append(opponent)
    winners = get_winner(players, table)
    # Return the winner == player
    return winners.__contains__(player)