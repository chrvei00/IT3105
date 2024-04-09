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
        self.players = util.rotate(players, dealer)
        self.deck = deck
        self.blind = blind
        self.pot = 0
        self.high_bet = 0
        self.table = []

    def __repr__(self):
        return f"Hand has {len(self.players)} players, the dealer is {self.dealer}, the pot is {self.pot}, the high bet is {self.high_bet}, and the table is {self.table}"
        
    def play(self):
        print("\nStarting new hand\n")
        print("\nShuffling deck, dealing cards, and posting blinds\n")
        self.deck.shuffle()
        self.deal_cards()
        self.blinds()
        round = 0
        gui.visualize_players(self.window, self.players)
        while not util.hand_over(self.players, self.table):
            self.deal_table(round)
            self.get_player_actions()
            round += 1
        print("\nHand over\n")
        # Show cards
        for player in self.players:
            print(f"{player.name} has {player.cards}")
        winners = util.get_winner(self.players, self.table)
        self.reward(winners)
        self.reset()

    def deal_cards(self):
        self.deck.shuffle()
        for i in range(2):
            for player in self.players:
                cards = self.deck.deal_card()
                for card in cards:
                    player.cards.append(card)
    
    def blinds(self):
        self.players[0].bet(self.blind)
        self.players[1].bet(self.blind * 2)
        self.pot += self.blind + self.blind * 2
        self.high_bet = self.blind * 2

    def fold(self, player: Player):
        self.players.remove(player)

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
        print(f"\nDealing {round_names[round]}\n")
        print(f"Table: {self.table}\n")
    
    def get_player_actions(self):
        first = True
        gui.visualize_players(self.window, self.players)
        while first or not util.round_over(self.players, self.high_bet):
            first = False
            for player in self.players:
                gui.visualize_players(self.window, self.players)
                action = player.get_action(self.window, self.high_bet, self.pot, self.table, self.players, self.blind)
                if action == "call":
                    amount = self.high_bet - player.current_bet
                    player.bet(amount)
                    self.pot += amount
                    self.high_bet = util.get_high_bet(self.players)
                elif action == "bet":
                    player.bet(self.blind * 2)
                    self.pot += self.blind * 2
                    self.high_bet = util.get_high_bet(self.players)
                elif action == "all-in":
                    amount = player.chips
                    player.bet(amount)
                    self.pot += amount
                    self.high_bet = util.get_high_bet(self.players)
                elif action == "fold":
                    self.players.remove(player)

    def reward(self, winners: list):
        for player in winners:
            player.reward(self.pot / len(winners))
        print(f"Winners: {winners}; Pot: {self.pot}")
        print(f"New chip counts: {[player for player in self.initial_players]}")

    def reset(self):
        for player in self.initial_players:
            player.reset_cards_and_bet()