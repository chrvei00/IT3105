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

# ... (continued) ...
