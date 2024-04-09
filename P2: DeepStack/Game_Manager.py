from Util.Hand import Hand
from Util.Card import Deck
import Util.Game_Util as util

class Game:
    def __init__(self, window, Num_Human_Players: int, Num_AI_Rollout_Players: int, Game_Type: str = "simple", start_chips: int = 1000):
        util.validate_game(Num_Human_Players, Num_AI_Rollout_Players, Game_Type)
        self.players, self.dealer, self.deck, self.blind = util.setup_game(Num_Human_Players, Num_AI_Rollout_Players, Game_Type, start_chips)
        self.window = window

    def __repr__(self):
        return f"Game has {len(self.players)} players, the dealer is {self.dealer}, and the blind is {self.blind}"

    def start(self):
        while not util.game_over(self.players):
            print("\n ----------------- \n")
            print(self)
            input("\nPress enter to start a new hand")
            # Play a hand
            hand = Hand(self.window, self.players, self.dealer, self.deck, self.blind)
            print(hand)
            hand.play()
            # Rotate the dealer and create a new deck
            self.dealer = util.next_dealer(self.players, self.dealer)
            self.deck = Deck()

        print(f"\nPlayer {self.players[0]} has won the game\n")