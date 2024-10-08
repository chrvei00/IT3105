import Game_Manager as Game_Manager
import Util.gui as gui
import Util.Config as config
import Util.N_Util as n_util
import Poker_Oracle as oracle
import inquirer

def play(auto):
    """
    Play a game of poker.

    Parameters:
    - auto (bool): Indicates whether the game should be played automatically or with user input.

    Returns:
    None
    """
    if auto:
        Num_Human_Players, Num_AI_Rollout_Players, Num_AI_Resolve_Players, start_chips = 2, 0, 0, 1000
    else:
        Num_Human_Players, Num_AI_Rollout_Players, Num_AI_Resolve_Players, start_chips = gui.create_game_setup_window()
    
    window = gui.create_poker_window(num_players=Num_Human_Players + Num_AI_Rollout_Players + Num_AI_Resolve_Players)
    game = Game_Manager.Game(window=window, Num_Human_Players=Num_Human_Players, Num_AI_Rollout_Players=Num_AI_Rollout_Players, Num_AI_Resolve_Players=Num_AI_Resolve_Players, start_chips=start_chips)
    game.start()
    window.close()

if __name__ == '__main__':
    questions = [
        inquirer.List('action',
                      message="What do you want to do?",
                      choices=['play', 'train', 'display UM'],
                  ),
    ]
    answers = inquirer.prompt(questions)
    if answers is None:
        exit()
    if answers.get('action') == 'train':
        n_util.train()
    elif answers.get('action') == 'display UM':
        oracle.display_utility_matrix()
    else:
        auto = config.read_setup()
        play(auto)
