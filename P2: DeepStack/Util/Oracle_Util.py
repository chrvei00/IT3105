import Util.Game_Util

def simulate(deck: object, table: list, player: object, opponents: list) -> bool:
    """
    Simulate a round of poker and return the winner of the hand.
    """
    deck.shuffle()
    if len(table) < 5:
        cards = deck.deal_card(5 - len(table))
        for card in cards:
            table.append(card)
    # Deal cards to opponents
    for opponent in opponents:
        cards = deck.deal_card(2)
        for card in cards:
            opponent.deal_card(card)
    # Determine the winner of the hand
    winners = Util.Game_Util.get_winner([player] + opponents, table)
    # Reset opponents hands
    for opponent in opponents:
        opponent.cards = []
    # Return the winner == player
    return winners.__contains__(player)

def represent_hand_as_string(hand: list) -> str:
    """
    Represent a hand as a string.
    """
    return f"{hand[0].__repr__()}, {hand[1].__repr__()}"