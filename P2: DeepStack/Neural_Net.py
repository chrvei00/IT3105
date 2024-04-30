import tensorflow as tf
import numpy as np
import datetime
import random
import Util.Card as Card

class Neural_Net:
    def __init__(self):
        self.model = tf.keras.models.load_model('nn/model_<function input_shape at 0x160786d40>_10_2024-04-30, 19:45.h5')
        self.model.compile()
    
    def evaluate(self, state, player_range, opponent_range):
        diff = 10 - len(encode_cards(state.table))
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
    encoded = []
    for card in cards:
        encoded.append(card.get_real_value())
        encoded.append(turn_suit_to_int(card.get_suit()))
    return encoded

def turn_suit_to_int(suit: str) -> int:
    if suit == "♥":
        return 0
    elif suit == "♦":
        return 1
    elif suit == "♣":
        return 2
    elif suit == "♠":
        return 3