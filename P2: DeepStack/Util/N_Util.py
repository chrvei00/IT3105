import Util.State_Util as state_util
import Util.Card as Card
import Util.Game_Util as game_util
import Util.resolver_util as resolver_util
import random
import numpy as np
import datetime
import tensorflow as tf

def input_shape() -> tuple[int]:
    """
    Returns the shape of the input for the neural network model.

    Returns:
        tuple[int]: The shape of the input.
    """
    player_range_dimension = len(state_util.gen_range())
    opponent_range_dimension = len(state_util.gen_range())
    public_cards_dimension = len(encode_cards(generate_public_cards()))
    pot_dimension = len([generate_relative_pot_size()])
    return (public_cards_dimension + pot_dimension + player_range_dimension + opponent_range_dimension,)

def output_shape() -> tuple[int]:
    """
    Returns the shape of the output for the neural network model.

    Returns:
        tuple[int]: The shape of the output.
    """
    player_range_dimension = len(state_util.gen_range())
    opponent_range_dimension = len(state_util.gen_range())
    return (player_range_dimension + opponent_range_dimension,)

def simulate_data(num_samples):
    """
    Simulates training data for the neural network model.

    Args:
        num_samples (int): The number of samples to generate.

    Returns:
        tuple[np.array, np.array]: The generated training features and targets.
    """
    train_features = []
    train_targets = []

    for _ in range(num_samples):
        public_cards = generate_public_cards()
        pot_size = generate_relative_pot_size()
        player1_range = state_util.gen_range()
        player2_range = state_util.gen_range()
        
        features = np.concatenate([
            np.array(list(player1_range.values())), 
            np.array(list(player2_range.values())), 
            encode_cards(public_cards),
            [0 for _ in range(10 - len(public_cards))],
            [pot_size]
        ])
        train_features.append(features)
        
        value_vector_1 = {hand: 0 for hand in player1_range}
        value_vector_2 = {hand: 0 for hand in player2_range}

        for hand in player1_range:
            value_vector_1[hand] = game_util.get_utility_potrelative(resolver_util.turn_hand_string_to_list(hand), public_cards, pot_size)
        for hand in player2_range:
            value_vector_2[hand] = game_util.get_utility_potrelative(resolver_util.turn_hand_string_to_list(hand), public_cards, pot_size)
        
        value_vector_1 = np.array(list(value_vector_1.values()))
        value_vector_2 = np.array(list(value_vector_2.values()))
        
        targets = np.concatenate([value_vector_1, value_vector_2])
        train_targets.append(targets)

    return np.array(train_features), np.array(train_targets)

def generate_public_cards() -> list:
    """
    Generates a list of public cards.

    Returns:
        list: The generated public cards.
    """
    deck = Card.Deck()
    deck.shuffle()
    return deck._deal(random.randint(0, 5))

def generate_relative_pot_size() -> float:
    """
    Generates a random relative pot size.

    Returns:
        float: The generated relative pot size.
    """
    return random.uniform(0, 2000)

def flatten_range(range: dict) -> list:
    """
    Flattens a range dictionary into a list.

    Args:
        range (dict): The range dictionary to flatten.

    Returns:
        list: The flattened range.
    """
    return [range[hand] for hand in range]

def encode_cards(cards: list) -> list:
    """
    Encodes a list of cards into a numerical representation.

    Args:
        cards (list): The list of cards to encode.

    Returns:
        list: The encoded cards.
    """
    encoded = []
    for card in cards:
        encoded.append(card.get_real_value())
        encoded.append(turn_suit_to_int(card.get_suit()))
    return encoded

def turn_suit_to_int(suit: str) -> int:
    """
    Converts a suit string to an integer representation.

    Args:
        suit (str): The suit string to convert.

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

def train():
    """
    Trains the neural network model.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape()),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(output_shape()[0])
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error')
    model.summary()
    num_samples = 100
    train_features, train_targets = simulate_data(num_samples)
    model.fit(train_features, train_targets, epochs=100, batch_size=32)

    date = datetime.datetime.now().strftime("%Y-%m-%d, %H:%M")
    model.save(f'nn/model_{input_shape()}_{num_samples}_{date}.h5')