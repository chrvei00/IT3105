from Util.Hand import Hand
from Util.Card import Deck
import Util.Game_Util as util
import Util.gui as gui

class Game:
    def __init__(self, window, Num_Human_Players: int, Num_AI_Rollout_Players: int, Num_AI_Resolve_Players: int=0, Game_Type: str = "simple", start_chips: int = 1000):
        util.validate_game(Num_Human_Players, Num_AI_Rollout_Players, Game_Type)
        self.players, self.dealer, self.deck, self.blind = util.setup_game(Num_Human_Players, Num_AI_Rollout_Players, Game_Type, start_chips)
        self.window = window

    def __repr__(self):
        return f"Game has {len(self.players)} players, the dealer is {self.dealer}, and the blind is {self.blind}"

    def start(self):
        gui.wait_for_user_to_start_new_hand_popup(self.window)
        while not util.game_over(self.players):
            # Display the players and their chips
            gui.visualize_players(self.window, self.players)
            # Play a hand
            hand = Hand(self.window, self.players, self.dealer, self.deck, self.blind)
            winner = hand.play()
            # Rotate the dealer and create a new deck
            self.dealer = util.next_dealer(self.players, self.dealer)
            self.deck = Deck()
            # Visualize winner
            gui.visualize_winner(winner)
            # Remove players with no chips
            for player in self.players:
                if player.chips <= self.blind * 2:
                    gui.add_history(self.window, f"{player.name} has run out of chips")
                    self.players.remove(player)
        # Display the winner
        gui.visualize_winner(f"\nPlayer {self.players[0]} has won the game\n")