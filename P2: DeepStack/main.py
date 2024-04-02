from Game_Manager import Game

def play():
    game = Game(Num_Human_Players=2, Num_AI_Players=0)
    game.start()

if __name__ == '__main__':
    play()