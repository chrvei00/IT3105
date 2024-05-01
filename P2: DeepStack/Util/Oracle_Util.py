import Util.Game_Util

def simulate(deck: object, table: list, player: object, opponents: list) -> bool:
    """
    Simulate a round of poker and return the winner of the hand.

    Parameters:
    - deck (object): The deck of cards used in the game.
    - table (list): The cards on the table.
    - player (object): The player object representing the player's hand.
    - opponents (list): A list of opponent objects representing their hands.

    Returns:
    - bool: True if the player is the winner, False otherwise.
    """
    deck.shuffle()
    if len(table) < 5:
        cards = deck.deal_card(5 - len(table))
        for card in cards:
            table.append(card)
    for opponent in opponents:
        cards = deck.deal_card(2)
        for card in cards:
            opponent.deal_card(card)
    winners = Util.Game_Util.get_winner([player] + opponents, table)
    for opponent in opponents:
        opponent.cards = []
    return winners.__contains__(player)

def represent_hand_as_string(hand: list) -> str:
    """
    Represent a hand as a string.

    Parameters:
    - hand (list): The hand to be represented.

    Returns:
    - str: The string representation of the hand.
    """
    return f"{hand[0].__repr__()}, {hand[1].__repr__()}"