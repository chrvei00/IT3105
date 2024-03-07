from Card import Card

class Player:
    def __init__(self, type: str = "human"):
        self.type = type
        self.chips = 0
        self.bet = 0
        self.cards = []
    
    def __repr__(self):
        return f"Player has {self.chips} chips"

    def deal_card(self, card: Card):
        print(f"Player has been dealt {card}")
        if len(self.cards) < 2:
            self.cards.append(card)
        else:
            raise ValueError("Player can only have 2 cards")
    
    def reward(self, amount: int):
        self.chips += amount
        print(f"Player has won {amount} chips")

    def bet(self, amount: int):
        if amount > self.chips:
            raise ValueError("Player cannot bet more than they have")
        self.chips -= amount
        print(f"Player has bet {amount} chips")

    def get_action(self, hand: Hand):
        if self.type == "human":
            action = input("Enter action (bet, call, fold): ")
            if action == "bet":
                amount = int(input("Enter amount: "))
                return ("bet", amount)
            elif action == "call":
                return ("call", hand.pot - self.bet)
            elif action == "fold":
                return ("fold", 0)
            else:
                raise ValueError("Action must be bet, call, or fold")
        else:
            return self.get_AI_action(game)

    def reset_cards_and_bet(self):
        self.cards = []
        self.bet = 0
