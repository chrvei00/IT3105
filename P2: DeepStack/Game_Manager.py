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
            self.deck = Deck()
            # Visualize winner
            gui.visualize_winner(winner)
            # Remove players with no chips
            prev_players = list(self.players)
            for player in self.players:
                if player.chips <= 0:
                    gui.add_history(self.window, f"{player.name} has run out of chips")
                    gui.remove_player(self.window, player)
                    self.players.remove(player)
            if len(self.players) == 1:
                break
            self.dealer = util.next_dealer(prev_players, self.dealer)
            while self.dealer not in self.players:
                self.dealer = util.next_dealer(prev_players, self.dealer)
            
        #Save the history and stats to file
        players_str = ', '.join([f"{player.name}: {player.chips}" for player in self.players])
        gui.save_history_to_file(self.window, players_str)
        # Display the winner
        gui.visualize_winner(f"\nPlayer {self.players[0]} has won the game\n")