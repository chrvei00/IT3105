import tensorflow as tf
import numpy as np
import datetime
import random
import Util.Card as Card

class Neural_Net:
    def __init__(self):
        """
        Initializes the Neural_Net class.
        Loads the pre-trained model and compiles it.
        """
        self.model = tf.keras.models.load_model('nn/model_(1099,)_100_2024-05-01, 23:45.h5')
        self.model.compile()
    
    def evaluate(self, state, player_range, opponent_range):
        """
        Evaluates the given state using the neural network model.

        Args:
            state (State): The current state of the game.
            player_range (dict): A dictionary representing the player's hand range.
            opponent_range (dict): A dictionary representing the opponent's hand range.

        Returns:
            player_value_vector (np.ndarray): The predicted value vector for the player's hand range.
            opponent_value_vector (np.ndarray): The predicted value vector for the opponent's hand range.
        """
        diff = 10 - len(encode_cards(state.table))
        if diff < 0:
            print(f"Error: Too many cards on the table. Expected 5, got {len(state.table)}")
        features = np.concatenate([
            np.array(list(player_range.values())),
            np.array(list(opponent_range.values())),
            encode_cards(state.table),
            [0 for _ in range(diff)],
            [sum(state.bets.values())]
        ])
        features = np.expand_dims(features, axis=0)
        value_vectors = self.model.predict(features)[0]
        player_value_vector = value_vectors[:len(player_range)]
        opponent_value_vector = value_vectors[len(player_range):]
        return player_value_vector, opponent_value_vector

def encode_cards(cards: list) -> list:
    """
    Encodes a list of cards into a numerical representation.

    Args:
        cards: A list of Card objects.

    Returns:
        encoded: A list of encoded card values and suits.
    """
    encoded = []
    for card in cards:
        encoded.append(card.get_real_value())
        encoded.append(turn_suit_to_int(card.get_suit()))
    return encoded

def turn_suit_to_int(suit: str) -> int:
    """
    Converts a suit string to its corresponding integer representation.

    Args:
        suit: A string representing the suit of a card.

    Returns:
        int: The integer representation of the suit.
    """
    if suit == "♥":
        return 0
    elif suit == "♦":
        return 1
    elif suit == "♣":
        return 2
    elif suit == "♠":
        return 3