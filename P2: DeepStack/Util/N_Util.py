import Util.State_Util as state_util
import Util.Card as Card
import Util.Game_Util as game_util
import Util.resolver_util as resolver_util
import random
import numpy as np
import datetime
import tensorflow as tf

def input_shape() -> tuple[int]:
    player_range_dimension = len(state_util.gen_range())
    opponent_range_dimension = len(state_util.gen_range())
    public_cards_dimension = len(encode_cards(generate_public_cards()))
    pot_dimension = len([generate_relative_pot_size()])
    return (public_cards_dimension + pot_dimension + player_range_dimension + opponent_range_dimension,)

def output_shape() -> tuple[int]:
    player_range_dimension = len(state_util.gen_range())
    opponent_range_dimension = len(state_util.gen_range())
    return (player_range_dimension + opponent_range_dimension,)

def simulate_data(num_samples):
    train_features = []
    train_targets = []

    for _ in range(num_samples):
        public_cards = generate_public_cards()
        pot_size = generate_relative_pot_size()
        player1_range = state_util.gen_range()
        player2_range = state_util.gen_range()
        
        # Flatten and concatenate features
        features = np.concatenate([
            np.array(list(player1_range.values())), 
            np.array(list(player2_range.values())), 
            encode_cards(public_cards), 
            [pot_size]
        ])
        train_features.append(features)
        
        # Initialize value vectors as empty lists or zeros array if needed
        value_vector_1 = {hand: 0 for hand in player1_range}
        value_vector_2 = {hand: 0 for hand in player2_range}

        for hand in player1_range:
            value_vector_1[hand] = game_util.get_utility_potrelative(resolver_util.turn_hand_string_to_list(hand), public_cards, pot_size)
        for hand in player2_range:
            value_vector_2[hand] = game_util.get_utility_potrelative(resolver_util.turn_hand_string_to_list(hand), public_cards, pot_size)
        
        # Convert dictionary values to arrays for concatenation
        value_vector_1 = np.array(list(value_vector_1.values()))
        value_vector_2 = np.array(list(value_vector_2.values()))
        
        # Concatenate arrays
        targets = np.concatenate([value_vector_1, value_vector_2])
        train_targets.append(targets)

    return np.array(train_features), np.array(train_targets)

def generate_public_cards() -> list:
    deck = Card.Deck()
    deck.shuffle()
    return deck._deal(5)

def generate_relative_pot_size() -> float:
    return random.uniform(0, 2000)

def flatten_range(range: dict) -> list:
    return [range[hand] for hand in range]

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

def train():
    # Define your neural network architecture
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape()),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(output_shape()[0])
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error')
    model.summary()
    # Assume `train_features` and `train_targets` are prepared according to your data generation step
    num_samples = 100
    train_features, train_targets = simulate_data(num_samples)
    model.fit(train_features, train_targets, epochs=100, batch_size=32)

    # Save the trained model with num_samples and date in the filename
    date = datetime.datetime.now().strftime("%Y-%m-%d, %H:%M")
    model.save(f'nn/model_{input_shape()}_{num_samples}_{date}.h5')