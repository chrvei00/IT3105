import PySimpleGUI as sg
import Util.Game_Util as util

def create_poker_window(num_players: int = 2):
    large_font = ('Helvetica', 20)
    player_rows = [
        [sg.Text(f"Player:", font=large_font), sg.Text('', key=f'-NAME-{i}-', size=(5, 1), font=large_font),
         sg.Text('Chips:', font=large_font), sg.Text('', key=f'-CHIPS-{i}-', size=(5, 1), font=large_font),
         sg.Text('Current Bet:', font=large_font), sg.Text('', key=f'-BET-{i}-', size=(5, 1), font=large_font)
        ]
        for i in range(num_players)
    ]
    layout = [
        [sg.Column(player_rows, key='-PLAYERS-', size=(500, 200), scrollable=False)],
        [sg.Text('', key='-INFO-', size=(50, 1), font=large_font)],
        [sg.Text('Table: ', font=large_font), sg.Text('', key='-TABLE-', size=(40, 1), font=large_font)],
        [sg.Text('Your cards: ', font=large_font), sg.Text('', key='-CARDS-', size=(40, 1), font=large_font)],
        [sg.Button('Call'), sg.Button('Raise'), sg.Button('Fold'), sg.Button('Refresh')]
    ]
    return sg.Window('Poker Game', layout, finalize=True)

def visualize_players(window, players: list):
    for idx, player in enumerate(players):
        window[f'-NAME-{idx}-'].update(player.name)
        window[f'-CHIPS-{idx}-'].update(str(player.chips))
        window[f'-BET-{idx}-'].update(str(player.current_bet))

def visualize_AI(window, table: list, name: str, chips: int, pot: int, current_bet: int, high_bet: int):
    info = f"{name} has {chips} chips, the pot is {pot}, the current bet is {current_bet}, and the high bet is {high_bet}"
    table_str = ', '.join(util.get_string_representation_cards(table))
    window['-INFO-'].update(info)
    window['-TABLE-'].update(table_str)
    window['-CARDS-'].update('')  # Clear the cards for AI's turn

def visualize_human(window, table: list, cards: list, name: str, chips: int, pot: int, current_bet: int, high_bet: int):
    info = f"{name}, press enter to choose your action.\n{name} has {chips} chips, the pot is {pot}, the current bet is {current_bet}, and the high bet is {high_bet}. To call: {high_bet - current_bet}"
    table_str = ', '.join(util.get_string_representation_cards(table))
    cards_str = ', '.join(util.get_string_representation_cards(cards))
    window['-INFO-'].update(info)
    window['-TABLE-'].update(table_str)
    window['-CARDS-'].update(cards_str)