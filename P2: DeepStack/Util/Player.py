import inquirer
from Util.Card import Card
from Poker_Oracle import hole_card_rollout
import Util.Game_Util as util
import random

class Player:
    def __init__(self, name: str, type: str = "human"):
        self.name = name
        self.type = type
        self.chips = 0
        self.current_bet = 0
        self.cards = []
        self.active_in_hand = True
        self.is_all_in = False
    
    def __repr__(self):
        return f"Player: {self.name} C: {self.chips}"

    def get_cards(self) -> list:
        return self.cards

    def deal_card(self, card: Card):
        if len(self.cards) < 2:
            self.cards.append(card)
        else:
            raise ValueError("Player can only have 2 cards")
    
    def reward(self, amount: int):
        self.chips += amount

    def bet(self, amount: int):
        if amount >= self.chips:
            self.is_all_in = True
            self.current_bet += self.chips
            self.chips = 0
            return (f"{self.name} is all in")
        self.chips -= amount
        self.current_bet += amount
        return (f"{self.name} has bet {amount} chips")

    def get_action(self, window, high_bet: int, pot: int, table: list, players: list, blind: int):
        if self.type == "human":
            return util.visualize_human(window, table, self.cards, self.name, self.chips, pot, self.current_bet, high_bet, actions=self.get_possible_actions(high_bet, blind))
        elif self.type == "AI-resolver":
            return self.get_AI_Resolver_action(high_bet, pot, table, players, blind)
        else:
            util.visualize_AI(window, table, self.name, self.chips, pot, self.current_bet, high_bet)
            return self.get_AI_Rollout_action(high_bet, pot, table, players, blind)

        return "fold"
    
    def get_possible_actions(self, high_bet: int, blind: int):
        actions = []
        if high_bet - self.current_bet >= 0:
            actions.append("call")
        if self.chips > blind * 2:
            actions.append("bet")
        actions.append("fold")
        return actions

    def reset_cards_and_bet(self):
        self.cards = []
        self.current_bet = 0
        self.active_in_hand = True

    def get_AI_Resolver_action(self, high_bet: int, pot: int, table: list, players: list, blind: int):
        pass

    def get_AI_Rollout_action(self, high_bet: int, pot: int, table: list, players: list, blind: int):
        payout = (hole_card_rollout(table, self.cards, len(players)-1, cache=False, save=False) * pot)/high_bet   
        pa = self.get_possible_actions(high_bet, blind)     
        if random.random() < 0.15:
            if len(pa) == 1:
                return pa[0]
            if pa.__contains__("fold"):
                pa.remove("fold")
            return random.choice(pa)
        else:
            if payout < 0.8:
                if high_bet - self.current_bet <= 0:
                    return "call"
                return "fold"
            elif payout < 1.4 and pa.__contains__("call"):
                return "call"
            elif payout > 1.4 and pa.__contains__("bet"):
                return "bet"
            else:
                return "fold"
        


