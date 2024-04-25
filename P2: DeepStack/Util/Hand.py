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
        self.all_in_pots = []
        round = 0
        gui.visualize_players(self.window, self.players)
        while not util.hand_over(self.active_players(), self.table):
            self.deal_table(round)
            self.get_player_actions(first_round=round==0)
            round += 1
        gui.add_history(self.window, "Hand over")
        if any(x.is_all_in for x in self.players):
            self.pot = util.adjust_hand_params(self.players, self.pot)
        
        winners = self.determine_winners()
        winner_str = ', '.join([f"{winner.name}: {winner.cards}" for winner in winners])
        self.reset()
        return winner_str
    
    def determine_winners(self):
        winners = []
        # Determine wether there is a difference in the bets of players (meaning some players are all in)
        while len(list(filter(lambda x: x.current_bet != self.high_bet, self.active_players()))) > 0:
            # Find the player with the lowest bet
            lowest_bet = min(filter(lambda x: x.current_bet != self.high_bet, self.active_players()), key=lambda x: x.current_bet)
            # Determine winners
            winners = util.get_winner(self.active_players(), self.table)
            for player in self.active_players():
                print(player.name, player.cards)
            print("Determining winners", winners, "line 57")
            # Remove the players matching the lowest bet from the active players
            self.pot -= lowest_bet.current_bet*(len(self.active_players()))
            # Reward winner and adjust pot and bets
            self.reward(winners, amount=lowest_bet.current_bet*len(self.active_players()))
            self.pot -= lowest_bet.current_bet*len(self.active_players())
            for player in self.active_players():
                if player.current_bet <= lowest_bet.current_bet:
                    player.active_in_hand = False
            for player in self.active_players():
                player.current_bet -= lowest_bet.current_bet

        if self.pot > 0:
            additional_winners = (util.get_winner(self.active_players(), self.table))
            for winner in additional_winners:
                winners.append(winner)
            print("Determining winners", additional_winners, "line 73")
            for winner in additional_winners:
                self.reward(additional_winners, amount=self.pot)
        return winners

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
        if self.players[0].chips == 0:
            self.players[0].is_all_in = True
            gui.add_history(self.window, f"{self.players[0].name} is all in")
        if self.players[1].chips == 0:
            self.players[1].is_all_in = True
            gui.add_history(self.window, f"{self.players[1].name} is all in")
        self.pot += self.players[0].current_bet + self.players[1].current_bet
        self.high_bet = util.get_high_bet(self.players)

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
    
    def get_player_actions(self, first_round: bool):
        gui.visualize_players(self.window, self.players)
        self.player_action_round(first_round)
        while not util.round_over(self.active_players(), self.high_bet):
            self.player_action_round()
        for player in self.players:
            player.has_raised = False

        
    def player_action_round(self, first_round: bool=False):
        if first_round:
            players = self.active_players()[2:] + self.active_players()[:2]
        else:
            players = self.active_players()
        for player in players:
            if(util.end_action_round(self.active_players())):
                break
            gui.update_turn(self.window, player, self.players)
            if player.is_all_in or (player.has_raised and player.current_bet == self.high_bet):
                continue
            gui.visualize_players(self.window, self.players)
            allowed_actions = player.get_possible_actions(self.high_bet, self.blind)
            action = player.get_action(self.window, self.high_bet, self.pot, self.table, self.active_players(), self.blind)
            if action not in allowed_actions:
                raise ValueError(f"Action {action} not allowed for player {player.name}")
            if action == "fold":
                self.fold(player)
            elif action == "call":
                prev_bet = player.current_bet
                gui.add_history(self.window, player.bet(self.high_bet - player.current_bet))
                self.pot += player.current_bet - prev_bet
                self.high_bet = util.get_high_bet(self.players)
            elif action == "bet":
                player.has_raised = True
                prev_bet = player.current_bet
                gui.add_history(self.window, player.bet(self.high_bet - player.current_bet + self.blind * 2))
                self.pot += player.current_bet - prev_bet
                self.high_bet = util.get_high_bet(self.players)
            elif action == "all-in":
                player.has_raised = True
                player.is_all_in = True
                prev_bet = player.current_bet
                gui.add_history(self.window, player.bet(player.chips))
                self.pot += player.current_bet - prev_bet
                self.high_bet = util.get_high_bet(self.players)



    def reward(self, winners: list, amount: int=None):
        if amount is not None:
            for player in winners:
                player.reward(amount)
        else:
            for player in winners:
                player.reward(self.pot / len(winners))
        gui.add_history(self.window, f"Winner(s): {', '.join([winner.name for winner in winners])}, amount: {amount}")

    def reset(self):
        for player in self.initial_players:
            player.reset_cards_and_bet()