import Poker_Oracle as oracle
import Util.Game_Util as util
import Util.Card as Card
import random
import Resolver as res
import Util.Node as node

class Player:
    def __init__(self, name: str, type: str = "human", index: int = 0):
        self.name = name
        self.type = type
        self.chips = 0
        self.current_bet = 0
        self.cards = []
        self.active_in_hand = True
        self.is_all_in = False
        self.has_raised = False
        self.index = index
        if type == "AI_resolve":
            self.player_range, self.opponent_range = util.generate_ranges()
    
    def __repr__(self):
        return f"Player: {self.name} C: {self.chips}"

    def get_cards(self) -> list:
        return self.cards

    def deal_card(self, card: Card.Card):
        if len(self.cards) < 2:
            self.cards.append(card)
        else:
            raise ValueError("Player can only have 2 cards")
    
    def reward(self, amount: int):
        self.chips += amount

    def bet(self, amount: int):
        if amount < 0:
            raise ValueError("Amount must be greater than 0")
        elif self.chips < 0:
            raise ValueError("Player does not have any chips")
        if amount >= self.chips:
            amount = self.chips
            self.is_all_in = True
            self.current_bet += self.chips
            self.chips = 0
            return (f"{self.name} is all in, with a bet of {amount} chips")
        self.chips -= amount
        self.current_bet += amount
        return (f"{self.name} has bet {amount} chips")

    def get_action(self, window, high_bet: int, pot: int, table: list, players: list, blind: int):
        if self.type == "human":
            return util.visualize_human(window, table, self.cards, self.name, self.chips, pot, self.current_bet, high_bet, actions=self.get_possible_actions(high_bet, blind))
        elif self.type == "AI_resolve":
            return self.get_AI_Resolver_action(high_bet, pot, table, players, blind)
        else:
            util.visualize_AI(window, table, self.name, self.chips, pot, self.current_bet, high_bet)
            return self.get_AI_Rollout_action(high_bet, pot, table, players, blind)

        return "fold"
    
    def get_possible_actions(self, high_bet: int, blind: int):
        actions = ["fold"]
        if high_bet - self.current_bet >= 0 and self.chips >= high_bet - self.current_bet:
            actions.append("call")
        if self.chips >= high_bet - self.current_bet + blind * 2 and self.has_raised == False:
            actions.append("bet")
        if self.chips > 0:
            actions.append("all-in")
        return actions

    def reset_cards_and_bet(self):
        self.cards = []
        self.current_bet = 0
        self.active_in_hand = True
        self.is_all_in = False
        self.has_raised = False

    def get_AI_Resolver_action(self, high_bet: int, pot: int, table: list, players: list, blind: int):
        # Generate the state
        state_type = "decision"
        bets = {}
        player_stacks = {}
        has_raised = {}
        has_called = {}
        for player in players:
            bets[player.name] = player.current_bet
            player_stacks[player.name] = player.chips
            has_raised[player.name] = player.has_raised
            has_called[player.name] = False
        
        state = node.State("decision", bets, blind, player_stacks, table, self.name, has_raised, has_called)
        return res.get_action(self, state)

    def get_AI_Rollout_action(self, high_bet: int, pot: int, table: list, players: list, blind: int):
        payout = (oracle.hole_card_rollout(table, self.cards, len(players)-1, cache=False, save=False) * pot)/high_bet   
        pa = self.get_possible_actions(high_bet, blind)     
        if random.random() < 0.15:
            if len(pa) == 1:
                return pa[0]
            if pa.__contains__("fold"):
                pa.remove("fold")
            if random.random() < 0.1 and pa.__contains__("all-in"):
                return "all-in"
            if pa.__contains__("all-in"):
                pa.remove("all-in")
            if len(pa) == 0:
                return "fold"
            return random.choice(pa)
        else:
            if payout < 0.8:
                if high_bet - self.current_bet <= 0:
                    return "call"
                return "fold"
            elif payout < 1.4 and pa.__contains__("call"):
                return "call"
            elif 2.8 > payout > 1.4 and pa.__contains__("bet"):
                return "bet"
            elif payout >= 2.8 and pa.__contains__("all-in"):
                return "all-in"
            else:
                return "fold"
        


