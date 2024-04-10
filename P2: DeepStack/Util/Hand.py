from Util.Card import Deck
import Util.Game_Util as util
from Util.Player import Player
import Util.gui as gui

class Hand:
    def __init__(self, window, players: list, dealer: Player, deck: Deck, blind: int):
        util.validate_hand(players, dealer, deck, blind)
        self.window = window
        self.dealer = dealer
        self.initial_players = list(players)
        self.players = list(players)
        self.players = util.rotate(self.players, dealer)
        self.deck = deck
        self.blind = blind
        self.pot = 0
        self.high_bet = 0
        self.table = []

    def __repr__(self):
        return f"Hand has {len(self.players)} players, the dealer is {self.dealer}, the pot is {self.pot}, the high bet is {self.high_bet}, and the table is {self.table}"
    
    def active_players(self):
        return [player for player in self.players if player.active_in_hand]

    def play(self):
        gui.add_history(self.window, "Starting new hand")
        self.deck.shuffle()
        self.deal_cards()
        self.blinds()
        round = 0
        gui.visualize_players(self.window, self.players)
        while not util.hand_over(self.active_players(), self.table):
            self.deal_table(round)
            self.get_player_actions()
            round += 1
        gui.add_history(self.window, "Hand over")
        # Show cards
        all_winners = []
        for player in self.active_players():
            gui.add_history(self.window, f"{player.name} has {player.cards}")
        winners = util.get_winner(self.active_players(), self.table)
        while len([winner for winner in winners if winner.is_all_in]) > 0:
            winners = util.get_winner(self.active_players(), self.table)        
            if len([winner for winner in winners if winner.is_all_in]) > 0:
                all_in_winners = sort([winner for winner in winners if winner.is_all_in], key=lambda x: x.current_bet)
                dealtmoney = 0
                for winner in all_in_winners:
                    all_winners.append(winner)
                    self.reward([winner], amount=(winner.current_bet-dealtmoney)*len(self.active_players()))
                    self.pot -= (winner.current_bet-dealtmoney)*len(self.active_players())
                    dealtmoney = winner.current_bet
                    winner.active_in_hand = False
        if self.pot > 0:
            winners = util.get_winner(self.active_players(), self.table)
            for winner in winners:
                all_winners.append(winner)
                self.reward([winner], amount=self.pot/len(winners))

        winner_str = ', '.join([f"{winner.name}: {winner.cards}" for winner in all_winners])
        self.reset()
        return winner_str

    def deal_cards(self):
        self.deck.shuffle()
        for i in range(2):
            for player in self.players:
                cards = self.deck.deal_card()
                for card in cards:
                    player.cards.append(card)
    
    def blinds(self):
        gui.add_history(self.window, f"{self.players[0].name} is the small blind and {self.players[1].name} is the big blind")
        self.players[0].bet(self.blind)
        self.players[1].bet(self.blind * 2)
        self.pot += self.blind + self.blind * 2
        self.high_bet = self.blind * 2

    def fold(self, player: Player):
        player.active_in_hand = False
        gui.add_history(self.window, f"{player.name} has folded")

    def deal_table(self, round: int):
        if round > 3:
            raise ValueError("Round cannot be greater than 3")
        if round == 0:
            return
        elif round == 1:
            cards = self.deck.deal_card(3)
        else:
            cards = self.deck.deal_card()
        for card in cards:
            self.table.append(card)
        round_names = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}
        gui.add_history(self.window, f"Dealing {round_names[round]}: {cards}")
    
    def get_player_actions(self):
        first = True
        gui.visualize_players(self.window, self.players)
        previous_raiser = None
        while first or not util.round_over(self.active_players(), self.high_bet):
            for player in self.active_players():
                gui.update_turn(self.window, player)
                if player.is_all_in:
                    continue
                if player == previous_raiser and player.current_bet == self.high_bet:
                    continue
                gui.visualize_players(self.window, self.players)
                action = player.get_action(self.window, self.high_bet, self.pot, self.table, self.active_players(), self.blind)
                if action == "call":
                    prev_bet = player.current_bet
                    gui.add_history(self.window, player.bet(self.high_bet - player.current_bet))
                    self.pot += player.current_bet - prev_bet
                    self.high_bet = util.get_high_bet(self.players)
                elif action == "bet":
                    previous_raiser = player
                    prev_bet = player.current_bet
                    gui.add_history(self.window, player.bet(self.high_bet - player.current_bet + self.blind * 2))
                    self.pot += player.current_bet - prev_bet
                    self.high_bet = util.get_high_bet(self.players)
                elif action == "all-in":
                    prev_bet = player.current_bet
                    gui.add_history(self.window, player.bet(player.chips))
                    self.pot += player.current_bet - prev_bet
                    self.high_bet = util.get_high_bet(self.players)
                elif action == "fold":
                    self.fold(player)
            first = False

    def reward(self, winners: list, amount: int=None):
        if amount is not None:
            for player in winners:
                player.reward(amount)
        else:
            for player in winners:
                player.reward(self.pot / len(winners))
        gui.add_history(self.window, f"Winner(s): {', '.join([winner.name for winner in winners])}")

    def reset(self):
        for player in self.initial_players:
            player.reset_cards_and_bet()