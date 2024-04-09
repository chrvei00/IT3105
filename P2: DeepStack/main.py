from Game_Manager import Game
from Util.gui import create_poker_window

def play():
    window = create_poker_window(4)
    window.read(timeout=0)
    game = Game(window=window, Num_Human_Players=1, Num_AI_Rollout_Players=3)
    game.start()
    window.close()

if __name__ == '__main__':
    play()