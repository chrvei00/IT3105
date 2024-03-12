from Util.Card import Deck
import Util.Game_Util as util
from Util.Player import Player

class Hand:
    def __init__(self, players: list, dealer: Player, deck: Deck, blind: int):
        util.validate_hand(players, dealer, deck, blind)
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
        while not util.hand_over(self.players, self.table):
            self.deal_table(round)
            self.get_player_actions()
            round += 1
        print("\nHand over\n")
        winners = util.get_winner(self.players, self.table)
        self.reward(winners)
        self.reset()

    def deal_cards(self):
        self.deck.shuffle()
        for i in range(2):
            for player in self.players:
                player.deal_card(self.deck.deal_card())
    
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
            for i in range(3):
                self.table.append(self.deck.deal_card())
        else:
            self.table.append(self.deck.deal_card())
        round_names = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}
        print(f"\nDealing {round_names[round]}")
        print(f"Table: {self.table}\n")
    
    def get_player_actions(self):
        first = True
        while first or not util.round_over(self.players, self.high_bet):
            first = False
            for player in self.players:
                action = player.get_action(self.high_bet, self.pot, self.table, self.players, self.blind)
                if action[0] == "bet" or action[0] == "call":
                    player.bet(action[1])
                    self.pot += action[1]
                    self.high_bet = util.get_high_bet(self.players)
                elif action[0] == "fold":
                    self.players.remove(player)

    def reward(self, winners: list):
        for player in winners:
            player.reward(self.pot / len(winners))
        print(f"Winners: {winners}; Pot: {self.pot}")
        print(f"New chip counts: {[player for player in self.initial_players]}")

    def reset(self):
        for player in self.initial_players:
            player.reset_cards_and_bet()