import random
import Util.Config as config

class Card:
    def __init__(self, suit: str, value: int):
        """
        Initialize a Card object with a given suit and value.

        Args:
            suit (str): The suit of the card (Hearts, Diamonds, Clubs, Spades).
            value (int): The value of the card (2-10, 11 for Jack, 12 for Queen, 13 for King, 14 for Ace).

        Raises:
            ValueError: If the suit or value is invalid.
        """
        if suit not in ["Hearts", "Diamonds", "Clubs", "Spades"]:
            raise ValueError("Invalid suit")
        if int(value) not in config.get_cards():
            raise ValueError("Invalid value", value)
        self.suit = suit
        self.value = value

    def __repr__(self):
        """
        Return a string representation of the card.

        Returns:
            str: The string representation of the card.
        """
        return f"{self.represent_suit(self.suit)}{self.represent_value(self.value)}"

    def represent_suit(self, suit: str):
        """
        Return the symbol representation of a given suit.

        Args:
            suit (str): The suit of the card.

        Returns:
            str: The symbol representation of the suit.
        """
        if suit == "Hearts":
            return "♥"
        elif suit == "Diamonds":
            return "♦"
        elif suit == "Clubs":
            return "♣"
        elif suit == "Spades":
            return "♠"

    def represent_value(self, value: int):
        """
        Return the string representation of a given value.

        Args:
            value (int): The value of the card.

        Returns:
            str: The string representation of the value.
        """
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
        """
        Return the string representation of the card's value.

        Returns:
            str: The string representation of the card's value.
        """
        return self.represent_value(self.value)

    def get_real_value(self) -> int:
        """
        Return the actual value of the card.

        Returns:
            int: The actual value of the card.
        """
        return self.value

    def get_suit(self):
        """
        Return the symbol representation of the card's suit.

        Returns:
            str: The symbol representation of the card's suit.
        """
        return self.represent_suit(self.suit)

    @staticmethod
    def get_all_cards():
        """
        Return a list of all possible cards.

        Returns:
            list: A list of Card objects representing all possible cards.
        """
        return [Card(suit, value) for suit in ["Hearts", "Diamonds", "Clubs", "Spades"] for value in config.get_cards()]


class Deck:
    def __init__(self):
        """
        Initialize a Deck object with a full deck of cards.
        """
        self.cards = [Card(suit, value) for suit in ["Hearts", "Diamonds", "Clubs", "Spades"] for value in config.get_cards()]

    def __repr__(self):
        """
        Return a string representation of the deck.

        Returns:
            str: The string representation of the deck.
        """
        return f"Deck of {self.count()} cards"

    def count(self):
        """
        Return the number of cards in the deck.

        Returns:
            int: The number of cards in the deck.
        """
        return len(self.cards)

    def _deal(self, num):
        """
        Deal a specified number of cards from the deck.

        Args:
            num (int): The number of cards to deal.

        Returns:
            list: A list of Card objects representing the dealt cards.

        Raises:
            ValueError: If all cards have been dealt.
        """
        count = self.count()
        actual = min([count, num])
        if count == 0:
            raise ValueError("All cards have been dealt")
        cards = self.cards[-actual:]
        self.cards = self.cards[:-actual]
        return cards

    def deal_card(self, num=1):
        """
        Deal a specified number of cards from the deck.

        Args:
            num (int, optional): The number of cards to deal. Defaults to 1.

        Returns:
            list: A list of Card objects representing the dealt cards.

        Raises:
            ValueError: If all cards have been dealt.
        """
        return self._deal(num)

    def deal_hand(self, hand_size):
        """
        Deal a hand of cards with a specified size from the deck.

        Args:
            hand_size (int): The size of the hand to deal.

        Returns:
            list: A list of Card objects representing the dealt hand.

        Raises:
            ValueError: If all cards have been dealt.
        """
        return self._deal(hand_size)

    def shuffle(self):
        """
        Shuffle the cards in the deck.
        """
        random.shuffle(self.cards)