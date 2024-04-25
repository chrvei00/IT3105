import PySimpleGUI as sg
import Util.Game_Util as util
import datetime
import os
import threading

def create_game_setup_window():
    layout = [
        [sg.Text('Number of human players:'),
            sg.InputText('1', key='-NUM_HUMAN_PLAYERS-', size=(5, 1))],
        [sg.Text('Number of AI Rollout players:'),
            sg.InputText('1', key='-NUM_AI_Rollout_PLAYERS-', size=(5, 1))],
        [sg.Text('Number of AI Resolver players:'),
            sg.InputText('0', key='-NUM_AI_Resolver_PLAYERS-', size=(5, 1))],
        [sg.Text('Game type:'),
            sg.InputText('simple', key='-GAME_TYPE-', size=(10, 1))],
        [sg.Text('Starting chips:'),
            sg.InputText('1000', key='-START_CHIPS-', size=(5, 1))],
        [sg.Submit()]
    ]
    while True:
        event, values = sg.Window('Game Setup', layout, font=('Helvetica', 20)).read(close=True)
        if event == sg.WIN_CLOSED:
            break
        if event == 'Submit':
            return int(values['-NUM_HUMAN_PLAYERS-']), int(values['-NUM_AI_Rollout_PLAYERS-']), int(values['-NUM_AI_Resolver_PLAYERS-']), values['-GAME_TYPE-'], int(values['-START_CHIPS-'])
    raise ValueError("Window closed before submitting")

def create_poker_window(num_players: int = 2):
    large_font = ('Helvetica', 20)
    player_rows = [
        [
         sg.Text(f"Player:"), sg.Text('', key=f'-NAME-{i}-', size=(5, 1)),
         sg.Text('Chips:'), sg.Text('', key=f'-CHIPS-{i}-', size=(5, 1)),
         sg.Text('Current Bet:'), sg.Text('', key=f'-BET-{i}-', size=(5, 1))
        ]
        for i in range(num_players)
    ]
    left_layout = [
        [sg.Text('Turn: '), sg.Text('', key='-TURN-', size=(40, 1))],
        [sg.Column(player_rows, key='-PLAYERS-', size=(500, 200), scrollable=False)],
        [sg.Text('Table: '), sg.Text('', key='-TABLE-', size=(40, 1))],
        [sg.Text('Your cards: '), sg.Text('', key='-CARDS-', size=(40, 1))],
        [sg.Text('', key='-INFO-', size=(50, 1))],
        [sg.Button('call'), sg.Button('bet'), sg.Button('fold')]
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
    # Layout for the popup window
    layout = [
        [sg.Text(message)],
        [sg.Button('OK', key='OK')]  # Use a specific key for the button
    ]

    # Create the window
    window = sg.Window('Popup', layout, modal=True, return_keyboard_events=True)

    # Event loop
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'OK', '\r', 'Return:603979789'):  # Check for return key ('\r') on Mac
            break

    window.close()

def visualize_players(window, players: list):
    for idx, player in enumerate(players):
        window[f'-NAME-{idx}-'].update(player.name)
        window[f'-CHIPS-{idx}-'].update(str(player.chips))
        window[f'-BET-{idx}-'].update(str(player.current_bet))

def visualize_AI(window, table: list, name: str, chips: int, pot: int, current_bet: int, high_bet: int):
    window['call'].update(visible=False)
    window['bet'].update(visible=False)
    window['fold'].update(visible=False)
    info = f"{name} has {chips} chips, the pot is {pot}, the current bet is {current_bet}, and the high bet is {high_bet}"
    table_str = ', '.join(util.get_string_representation_cards(table))
    window['-INFO-'].update(info)
    window['-TABLE-'].update(table_str)
    window['-CARDS-'].update('')  # Clear the cards for AI's turn

def visualize_human(window, table: list, cards: list, name: str, chips: int, pot: int, current_bet: int, high_bet: int, actions: list):
    info = f"Pot: {pot}, your current bet: {current_bet}, highest bet: {high_bet}, to call: {high_bet - current_bet}"
    table_str = ', '.join(util.get_string_representation_cards(table))
    cards_str = ', '.join(util.get_string_representation_cards(cards))
    window['-INFO-'].update(info)
    window['-TABLE-'].update(table_str)
    window['-CARDS-'].update(cards_str)
    # Decide which buttons to show depending on the possible actions
    for action in ['call', 'bet', 'fold']:
        window[action].update(visible=action in actions)
    while True:
        event, values = window.read(timeout=None)
        if event == sg.WIN_CLOSED:
            break
        if event in actions or event in ['q', 'w', 'e']:
            # Translate from q, w, e to call, bet, fold
            if event == 'q':
                act = 'call'
            elif event == 'w':
                act = 'bet'
            elif event == 'e':
                act = 'fold'
            else:
                act = event
            break
    return act

def visualize_winner(window, winner: str):
    window['-INFO-'].update(f"{winner} has won the hand")
    window['-TABLE-'].update('')
    window['-CARDS-'].update('')
    window['call'].update(visible=False)
    window['bet'].update(visible=False)
    window['fold'].update(visible=False)
    custom_popup(f"{winner} has won the hand")

def wait_for_user_to_start_new_hand_popup(window):
    # Show a popup to wait for the user to start a new hand
    custom_popup("Press OK to start a new hand")

def visualize_winner(winner_str: str):
    custom_popup(f"{winner_str}")
    
def add_history(window, message):
    current_history = window['-HISTORY-'].get()
    new_message = message + "\n" + "-"*50 + "\n" + current_history  # Prepend new message and separator
    window['-HISTORY-'].update(value=new_message)

def update_turn(window, player):
    window['-TURN-'].update(f"{player.name} is up")

def save_history_to_file(window, players: str):
    # Get string representation of the current date with time
    date = datetime.datetime.now().strftime("%Y-%m-%d, %H:%M")
    history = window['-HISTORY-'].get()
    # Create a new file with the date as the name
    if not os.path.exists("./log"):
        os.mkdir("./log")
    with open(f"./log/{date}.txt", "w") as f:
        # Write the contents of the history variable to the file
        f.write(f"{date}\n{players}\n{history}")