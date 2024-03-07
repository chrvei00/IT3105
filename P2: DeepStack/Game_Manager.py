import util
from Player import Player
from Card import Card, Deck

class Hand:
    def __init__(self, players: list, dealer: Player, deck: Deck, blind: int):
        self.dealer = dealer
        self.players = util.rotate(players, self.players.index(self.dealer))
        self.deck = deck
        self.pot = 0
        self.table = []
        
    def play(self):
        self.deck.shuffle()
        self.deal_cards()
        self.blinds()
        self.validate_game_state()
        round = 0
        while not util.hand_over(this):
            self.deal_table(round)
            self.get_player_actions()
            round += 1
        self.reward()
        self.reset()

    def deal_cards(self):
        deck.shuffle()
        for i in range(2):
            for player in self.players:
                player.deal_card(self.deck.draw())
    
    def blinds(self):
        self.players[0].bet(self.blind)
        self.players[1].bet(self.blind * 2)

    def deal_table(self, round: int):
        if round > 3:
            raise ValueError("Round cannot be greater than 3")
        if round == 0:
            return
        elif round == 1:
            for i in range(3):
                self.table.append(self.deck.draw())
        else:
            self.table.append(self.deck.draw())
    
    def get_player_actions(self):
        while not util.round_over(self):
            for player in self.players:
                action = player.get_action(self)
                if action.type == "bet":
                    self.pot += action.amount
                elif action.type == "fold":
                    self.players.remove(player)

    def reward(self, player: Player):
        print(f"Player has won {self.pot} chips")
        player.reward(self.pot)

    def reset(self):
        for player in self.players:
            player.reset_cards_and_bet()

class Game:
    def __init__(self, Num_Human_Players: int, Num_AI_Players: int, Game_Type: str = "simple"):
        util.validate_game(Num_Human_Players, Num_AI_Players, Game_Type)
        self.players, self.dealer, self.deck = util.setup_game(Num_Human_Players, Num_AI_Players, Game_Type)

    def __repr__(self):
        return f"Game has {len(self.players)} players"

    def start(self):
        self.validate_game_state()
        while not self.game_over():
            self.dealer = util.next_dealer(self.players, self.dealer)
            Hand(self.players, self.dealer, self.deck).play()        