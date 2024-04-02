from Game_Manager import Game
from Poker_Oracle import hole_card_rollout, cheat_sheet
from Util.Card import Deck

def play():
    game = Game(Num_Human_Players=2, Num_AI_Players=0)
    game.start()
def sim():
    deck = Deck()
    table = []
    deck.shuffle()
    hand = deck.deal_hand(2)
    opponents = 1
    win_probability = hole_card_rollout(table, hand, opponents, deck)
    print(f"Win Probability: {hand} {win_probability}")

if __name__ == '__main__':
    # play()
    sim()