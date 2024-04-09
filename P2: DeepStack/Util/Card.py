import random

class Card:
    def __init__(self, suit: str, value: int):
        self.suit = suit
        self.value = value

    def __repr__(self):
        return f"{self.represent_suit(self.suit)}{self.represent_value(self.value)}"

    def represent_suit(self, suit: str):
        if suit == "Hearts":
            return "♥"
        elif suit == "Diamonds":
            return "♦"
        elif suit == "Clubs":
            return "♣"
        elif suit == "Spades":
            return "♠"
    def represent_value(self, value: int):
        if value == 11:
            return "J"
        elif value == 12:
            return "Q"
        elif value == 13:
            return "K"
        elif value == 14:
            return "A"
        else:
            return value
    def get_value(self):
        return self.represent_value(self.value)
    def get_real_value(self):
        return self.value
    def get_suit(self):
        return self.represent_suit(self.suit)

    def get_all_cards():
        return [Card(suit, value) for suit in ["Hearts", "Diamonds", "Clubs", "Spades"] for value in [2,3,4,5,6,7,8,9,10,11,12,13,14]]


class Deck:
    def __init__(self):
        self.cards = [Card(suit, value) for suit in ["Hearts", "Diamonds", "Clubs", "Spades"] for value in [2,3,4,5,6,7,8,9,10,11,12,13,14]]

    def __repr__(self):
        return f"Deck of {self.count()} cards"

    def count(self):
        return len(self.cards)

    def _deal(self, num):
        count = self.count()
        actual = min([count, num])
        if count == 0:
            raise ValueError("All cards have been dealt")
        cards = self.cards[-actual:]
        self.cards = self.cards[:-actual]
        return cards

    def deal_card(self, num=1):
        return self._deal(num)

    def deal_hand(self, hand_size):
        return self._deal(hand_size)

    def shuffle(self):
        random.shuffle(self.cards)