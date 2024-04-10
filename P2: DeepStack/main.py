from Game_Manager import Game
import Util.gui as gui

def play():
    Num_Human_Players, Num_AI_Rollout_Players, Num_AI_Resolve_Players, Game_Type, start_chips = gui.create_game_setup_window()
    window = gui.create_poker_window(num_players=Num_Human_Players + Num_AI_Rollout_Players + Num_AI_Resolve_Players)
    game = Game(window=window, Num_Human_Players=Num_Human_Players, Num_AI_Rollout_Players=Num_AI_Rollout_Players, Num_AI_Resolve_Players=0, Game_Type="simple", start_chips=start_chips)
    game.start()
    window.close()

if __name__ == '__main__':
    play()