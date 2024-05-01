import configparser

def read_setup() -> bool:
    """
    Read the setup from the config file.
    
    Returns:
        bool: The value of the 'auto' key in the 'setup' section of the config file.
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config.getboolean('setup', 'auto')

def read_end_depth() -> int:
    """
    Read the end depth from the config file.
    
    Returns:
        int: The value of the 'end_depth' key in the 'setup' section of the config file.
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config.getint('setup', 'end_depth')

def read_rollouts() -> int:
    """
    Read the number of rollouts from the config file.
    
    Returns:
        int: The value of the 'rollouts' key in the 'setup' section of the config file.
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config.getint('setup', 'rollouts')

def read_simultation_size() -> int:
    """
    Read the simulation size from the config file.
    
    Returns:
        int: The value of the 'simulation_size' key in the 'oracle' section of the config file.
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config.getint('oracle', 'simulation_size')

def read_cheat_sheet() -> dict:
    """
    Read the cheat sheet from the config file.
    
    Returns:
        dict: A dictionary containing the key-value pairs from the 'cheat_sheet' section of the config file.
              Returns None if the 'cheat_sheet' section does not exist.
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    if not config.has_section('cheat_sheet'):
        return None
    return dict(config['cheat_sheet'])

def read_blind() -> int:
    """
    Read the blind from the config file.
    
    Returns:
        int: The value of the 'blind' key in the 'setup' section of the config file.
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config.getint('setup', 'blind')

def read_monitor() -> bool:
    """
    Read the monitor from the config file.
    
    Returns:
        bool: The value of the 'monitor' key in the 'setup' section of the config file.
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config.getboolean('setup', 'monitor')

def read_nn_evalution() -> bool:
    """
    Read the neural network evaluation from the config file.
    
    Returns:
        bool: The value of the 'nn_evaluation' key in the 'setup' section of the config file.
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config.getboolean('setup', 'nn_evaluation')

def get_cards() -> list:
    """
    Return a list of all possible cards.
    
    Returns:
        list: A list of integers representing the possible cards.
    """
    return [2, 3, 4, 5, 6, 7, 8, 9]

def write_cheat_sheet(hand: list, opponents: int, wins: int, n: int) -> None:
    """
    Write the win rate of a hand to the cheat sheet.
    
    Args:
        hand (list): A list of cards representing the hand.
        opponents (int): The number of opponents.
        wins (int): The number of wins.
        n (int): The total number of games played.
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    if not config.has_section('cheat_sheet'):
        config.add_section('cheat_sheet')
    config.set("cheat_sheet", f"{format_hand(hand, opponents)}", f"{wins / n}")
    with open('config.ini', 'w') as configfile:
        config.write(configfile)
    return

def format_hand(hand: list, opponents: int = 1) -> str:
    """
    Return a formatted string of a hand.
    
    Args:
        hand (list): A list of cards representing the hand.
        opponents (int, optional): The number of opponents. Defaults to 1.
    
    Returns:
        str: A formatted string representing the hand.
    """
    hand.sort(key=lambda x: x.get_real_value(), reverse=True)
    format_hand = f"{hand[0].get_value()}{hand[1].get_value()}"
    if (hand[0].get_suit() == hand[1].get_suit()):
        suited = "suited"
    else:
        suited = "offsuit"
    return f"{format_hand}-{suited}-{opponents+1}".lower()

def format_hole_pair(pair: list) -> tuple:
    """
    Sort a hand by the value of the cards.
    
    Args:
        pair (list): A list of cards representing the hand.
    
    Returns:
        tuple: A tuple of strings representing the sorted hand.
    """
    pair.sort(key=lambda x: x.get_real_value(), reverse=True)
    pair.sort(key=lambda x: x.get_suit())
    string_pair = f"{pair[0].__repr__()}{pair[1].__repr__()}"
    return string_pair

def get_actions() -> list:
    """
    Return a list of possible actions.
    
    Returns:
        list: A list of strings representing the possible actions.
    """
    return ["fold", "call", "bet"]