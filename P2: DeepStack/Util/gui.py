import PySimpleGUI as sg
import Util.Game_Util as util
import datetime
import os
import threading
import Util.Config as config

def create_game_setup_window():
    """
    Creates a window for game setup.

    Returns:
        tuple: A tuple containing the number of human players, number of AI Rollout players,
               number of AI Resolver players, and starting chips.
    """
    layout = [
        [sg.Text('Number of human players:'),
            sg.InputText('1', key='-NUM_HUMAN_PLAYERS-', size=(5, 1))],
        [sg.Text('Number of AI Rollout players:'),
            sg.InputText('0', key='-NUM_AI_Rollout_PLAYERS-', size=(5, 1))],
        [sg.Text('Number of AI Resolver players:'),
            sg.InputText('1', key='-NUM_AI_Resolver_PLAYERS-', size=(5, 1))],
        [sg.Text('Starting chips:'),
            sg.InputText('1000', key='-START_CHIPS-', size=(5, 1))],
        [sg.Submit()]
    ]
    while True:
        event, values = sg.Window('Game Setup', layout, font=('Helvetica', 20)).read(close=True)
        if event == sg.WIN_CLOSED:
            break
        if event == 'Submit':
            return int(values['-NUM_HUMAN_PLAYERS-']), int(values['-NUM_AI_Rollout_PLAYERS-']), int(values['-NUM_AI_Resolver_PLAYERS-']), int(values['-START_CHIPS-'])
    raise ValueError("Window closed before submitting")

def create_poker_window(num_players: int = 2):
    """
    Creates a window for the poker game.

    Args:
        num_players (int): The number of players in the game. Defaults to 2.

    Returns:
        PySimpleGUI.Window: The created poker game window.
    """
    large_font = ('Helvetica', 20)
    player_rows = [
        [
         sg.Text(f"Player:"), sg.Text('', key=f'-NAME-{i}-', size=(10, 1)),
         sg.Text('Chips:'), sg.Text('', key=f'-CHIPS-{i}-', size=(5, 1)),
         sg.Text('Current Bet:'), sg.Text('', key=f'-BET-{i}-', size=(5, 1))
        ]
        for i in range(num_players)
    ]
    left_layout = [
        [sg.Text('Turn: '), sg.Text('', key='-TURN-', size=(40, 1))],
        [sg.Column(player_rows, key='-PLAYERS-', size=(700, 200), scrollable=False)],
        [sg.Text('Table: '), sg.Text('', key='-TABLE-', size=(40, 1))],
        [sg.Text('Your cards: '), sg.Text('', key='-CARDS-', size=(40, 1))],
        [sg.Text('', key='-INFO-', size=(50, 3))],
        [sg.Button('fold'), sg.Button('call'), sg.Button('bet'), sg.Button('all-in')]
    ]
    right_layout = [
        [sg.Text("History")],
        [sg.Multiline(key='-HISTORY-', size=(30, 20), disabled=True, autoscroll=False)]
    ]

    layout = [
        [
            sg.Column(left_layout), 
            sg.VSeperator(), 
            sg.Column(right_layout),
        ]
    ]

    return sg.Window('Poker Game', layout, finalize=True, font=large_font, return_keyboard_events=True)

def custom_popup(message):
    """
    Displays a custom popup window with a message.

    Args:
        message (str): The message to display in the popup window.
    """
    layout = [
        [sg.Text(message)],
        [sg.Button('OK', key='OK')]
    ]

    window = sg.Window('Popup', layout, modal=True, return_keyboard_events=True)

    while True:
        event, values = window.read(timeout=None)
        if event in (sg.WIN_CLOSED, 'OK', '\r', 'Return:603979789'):
            break

    window.close()

def visualize_players(window, players: list):
    """
    Updates the player information in the poker game window.

    Args:
        window (PySimpleGUI.Window): The poker game window.
        players (list): A list of player objects.
    """
    for player in players:
        window[f'-NAME-{player.index}-'].update(player.name)
        window[f'-CHIPS-{player.index}-'].update(str(player.chips))
        window[f'-BET-{player.index}-'].update(str(player.current_bet))

def visualize_AI(window, table: list, name: str, chips: int, pot: int, current_bet: int, high_bet: int):
    """
    Updates the GUI to visualize the AI's decision-making process during a poker game.

    Parameters:
    - window: The GUI window object.
    - table: A list of cards on the table.
    - name: The name of the AI player.
    - chips: The number of chips the AI player has.
    - pot: The current size of the pot.
    - current_bet: The current bet amount.
    - high_bet: The highest bet amount.

    Returns:
    None
    """
    for action in ['fold', 'call', 'bet', 'all-in']:
        window[action].update(visible=False)
    info = f"{name} is deciding what to do..."
    table_str = ', '.join(util.get_string_representation_cards(table))
    window['-INFO-'].update(info)
    window['-TABLE-'].update(table_str)
    window['-CARDS-'].update('')  # Clear the cards for AI's turn
    window.refresh()

def visualize_human(window, table: list, cards: list, name: str, chips: int, pot: int, current_bet: int, high_bet: int, actions: list):
    """
    Visualizes the game state for a human player and waits for the player to make a decision.

    Args:
        window: The GUI window object.
        table (list): The community cards on the table.
        cards (list): The player's hole cards.
        name (str): The player's name.
        chips (int): The player's chip count.
        pot (int): The current pot size.
        current_bet (int): The player's current bet.
        high_bet (int): The highest bet on the table.
        actions (list): The list of possible actions for the player.

    Returns:
        str: The action chosen by the player.

    """
    info = f"Pot: {pot}, your current bet: {current_bet}, highest bet: {high_bet}, to call: {high_bet - current_bet}"
    table_str = ', '.join(util.get_string_representation_cards(table))
    cards_str = ', '.join(util.get_string_representation_cards(cards))
    window['-INFO-'].update(info)
    window['-TABLE-'].update(table_str)
    window['-CARDS-'].update(cards_str)
    # Decide which buttons to show depending on the possible actions
    for action in config.get_actions():
        window[action].update(visible=True)
    # Wait for the user to press a button
    while True:
        event, values = window.read(timeout=None)
        if event == sg.WIN_CLOSED:
            break
        if event in ['q', 'w', 'e', 'r']:
            # Translate from q, w, e to call, bet, fold
            if event == 'q':
                act = 'fold'
            elif event == 'w':
                act = 'call'
            elif event == 'e':
                act = 'bet'
            elif event == 'r':
                act = 'all-in'
            else:
                act = event
            
            if act in actions:
                break
            else:
                custom_popup(f"Action {act} not allowed")
    return act

def visualize_winner(window, winner: str):
    """
    Updates the GUI window to visualize the winner of the game.

    Args:
        window: The GUI window object.
        winner (str): The name of the winner.

    Returns:
        None
    """
    window['-INFO-'].update(winner)
    window['-TABLE-'].update('')
    window['-CARDS-'].update('')
    window['call'].update(visible=False)
    window['bet'].update(visible=False)
    window['fold'].update(visible=False)
    if(config.read_monitor()):
        custom_popup(winner)
def wait_for_user_to_start_new_hand_popup(window):
    """
    Displays a popup window and waits for the user to start a new hand.

    Args:
        window: The main window object.

    Returns:
        None
    """
    # Show a popup to wait for the user to start a new hand
    if config.read_monitor():
        custom_popup("Press OK to start a new hand")
    
def add_history(window, message):
    """
    Adds a new message to the history in the GUI window.

    Parameters:
    - window: The GUI window object.
    - message: The message to be added to the history.

    Returns:
    None
    """
    current_history = window['-HISTORY-'].get()
    new_message = message + "\n" + "-"*50 + "\n" + current_history  # Prepend new message and separator
    window['-HISTORY-'].update(value=new_message)

def update_turn(window, player, players: list, timer: int = 0):
    """
    Updates the turn information in the GUI window.

    Parameters:
    - window: The GUI window object.
    - player: The current player object.
    - players: A list of all player objects.
    - timer: The timer value for the player's decision (default: 0).
    """
    window.refresh()
    if player.type == "human":
        window['-TURN-'].update(f"{player.name} is up (q: fold, w: call, e: bet, r: all-in)")
    else:
        window['-TURN-'].update(f"{player.name} is up")
    if player.type != "human":
        for action in ['fold', 'call', 'bet', 'all-in']:
            window[action].update(visible=False)
        window['-INFO-'].update(f"{player.name} is deciding what to do... timer: {timer}")
    for o_player in players:
        if not o_player.active_in_hand:
            window[f'-NAME-{o_player.index}-'].update(text_color='black', background_color='red')
        else:
            window[f'-NAME-{o_player.index}-'].update(text_color='black', background_color='white')
    window[f'-NAME-{player.index}-'].update(f'{player.name}', text_color='black', background_color='green')
    window.refresh()

def remove_player(window, player):
    """
    Removes a player from the GUI window.

    Parameters:
    - window: The GUI window object.
    - player: The player object to be removed.

    Returns:
    None
    """
    if window[f'-NAME-{player.index}-'].get() == player.name:
        window[f'-NAME-{player.index}-'].update(f'{player.name} (out)', text_color='red')
        window[f'-CHIPS-{player.index}-'].update('')
        window[f'-BET-{player.index}-'].update('')

def save_history_to_file(window, players: str):
    """
    Save the history of the game to a file.

    Parameters:
    - window: The GUI window object.
    - players: A string representing the players in the game.

    Returns:
    None
    """
    # Get string representation of the current date with time
    date = datetime.datetime.now().strftime("%Y-%m-%d, %H:%M")
    history = window['-HISTORY-'].get()
    # Create a new file with the date as the name
    if not os.path.exists("./log"):
        os.mkdir("./log")
    with open(f"./log/{date}.txt", "w") as f:
        # Write the contents of the history variable to the file
        f.write(f"{date}\n{players}\n{history}")