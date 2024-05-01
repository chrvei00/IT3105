import Util.Hand as Hand
import Util.Card as Card
import Util.Game_Util as util
import Util.gui as gui

class Game:
    """
    Represents a game of poker.

    Args:
        window: The GUI window for displaying the game.
        Num_Human_Players (int): The number of human players in the game.
        Num_AI_Rollout_Players (int): The number of AI players using rollout strategy in the game.
        Num_AI_Resolve_Players (int): The number of AI players using resolve strategy in the game.
        start_chips (int, optional): The number of starting chips for each player. Defaults to 1000.
    """

    def __init__(self, window, Num_Human_Players: int, Num_AI_Rollout_Players: int, Num_AI_Resolve_Players: int, start_chips: int = 1000):
        util.validate_game(Num_Human_Players, Num_AI_Rollout_Players, Num_AI_Resolve_Players)
        self.players, self.dealer, self.deck, self.blind = util.setup_game(Num_Human_Players, Num_AI_Rollout_Players, Num_AI_Resolve_Players, start_chips)
        self.window = window

    def __repr__(self):
        """
        Returns a string representation of the Game object.

        Returns:
            str: A string representation of the Game object.
        """
        return f"Game has {len(self.players)} players, the dealer is {self.dealer}, and the blind is {self.blind}"

    def start(self):
        """
        Starts the game.

        This method runs the main game loop until the game is over. It handles the logic for each hand of poker,
        including dealing cards, determining the winner, updating player chips, and managing the dealer position.
        """
        gui.wait_for_user_to_start_new_hand_popup(self.window)
        while not util.game_over(self.players):
            gui.visualize_players(self.window, self.players)
            hand = Hand.Hand(self.window, self.players, self.dealer, self.deck, self.blind)
            winner = hand.play()
            self.deck = Card.Deck()
            gui.visualize_winner(self.window, f"{winner} has won the hand")
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
            
        players_str = ', '.join([f"{player.name}: {player.chips}" for player in self.players])
        gui.save_history_to_file(self.window, players_str)
        gui.visualize_winner(self.window, f"\nPlayer {self.players[0]} has won the game\n")