def read_simultation_size() -> int:
    """
    Read the simulation size from the config file.
    """
    config = ConfigParser()
    config.read('config.ini')
    return config.getint('oracle', 'simulation_size')

def read_cheat_sheet() -> dict:
    """
    Read the cheat sheet from the config file.
    """
    config = ConfigParser()
    config.read('config.ini')
    if not config.has_section('cheat_sheet'):
        return None
    return dict(config['cheat_sheet'])

def write_cheat_sheet(hand: list, opponents: int, wins: int, n: int) -> None:
    """
    Write the win rate of a hand to the cheat sheet.
    """
    config = ConfigParser()
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
    """
    hand.sort(key=lambda x: x.get_real_value(), reverse=True)
    format_hand = f"{hand[0].get_value()}{hand[1].get_value()}"
    if (hand[0].get_suit() == hand[1].get_suit()):
        suited = "suited"
    else:
        suited = "offsuit"
    return f"{format_hand}-{suited}-{opponents+1}".lower()