import inquirer
from Util.Card import Card

class Player:
    def __init__(self, name: str, type: str = "human"):
        self.name = name
        self.type = type
        self.chips = 0
        self.current_bet = 0
        self.cards = []
    
    def __repr__(self):
        return f"Player: {self.name} C: {self.chips}"

    def deal_card(self, card: Card):
        if len(self.cards) < 2:
            self.cards.append(card)
        else:
            raise ValueError("Player can only have 2 cards")
    
    def reward(self, amount: int):
        self.chips += amount

    def bet(self, amount: int):
        if amount > self.chips:
            raise ValueError("Player cannot bet more than they have")
        self.chips -= amount
        self.current_bet += amount
        print(f"{self.name} has bet {amount} chips")

    def get_action(self, high_bet: int, pot: int, table: list, players: list, blind: int):
        print(f"\n{self.name}'s turn")
        print(f"Cards: {self.cards}")
        print(f"Table: {table}")
        print(f"Chips: {self.chips}")
        print(f"Pot: {pot}")
        print(f"Current bet: {self.current_bet}")
        print(f"Highest bet: {high_bet}")
        if self.type == "human":
            action = inquirer.list_input("Enter action", choices=[f"call ({high_bet - self.current_bet})", "bet", "fold"])
            if action == "bet":
                amount = inquirer.list_input(f"Enter amount: {high_bet-self.current_bet} + ", choices=[blind*2, blind*4])
                return ("bet", int(amount) + high_bet - self.current_bet)
            elif action == f"call ({high_bet - self.current_bet})":
                return ("call", high_bet - self.current_bet)
            elif action == "fold":
                return ("fold", 0)
            else:
                raise ValueError("Action must be bet, call, or fold")
        # else:
        #     return self.get_AI_action(game)
        return ("fold", 0)

    def reset_cards_and_bet(self):
        self.cards = []
        self.current_bet = 0
