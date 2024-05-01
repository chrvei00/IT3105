import Poker_Oracle as oracle
import Util.Game_Util as util
import Util.Card as Card
import random
import Resolver as res
import Util.Node as node
import Util.gui as gui
import threading
import time
import copy
import Util.Config as config

class Player:
    """
    Represents a player in a poker game.

    Attributes:
        name (str): The name of the player.
        type (str): The type of the player, can be "human" or "AI_resolve".
        index (int): The index of the player.
        chips (int): The number of chips the player has.
        current_bet (int): The current bet made by the player.
        cards (list): The cards held by the player.
        active_in_hand (bool): Indicates if the player is active in the current hand.
        is_all_in (bool): Indicates if the player is all-in.
        has_raised (bool): Indicates if the player has raised.
        action (str): The action chosen by the player.
        resolver (Resolver): The resolver used by the player for AI_resolve type.
        player_range (list): The range of hands for the player.
        opponent_range (list): The range of hands for the opponent.

    Methods:
        __init__(self, name: str, type: str = "human", index: int = 0): Initializes a Player object.
        __repr__(self): Returns a string representation of the Player object.
        get_cards(self) -> list: Returns the cards held by the player.
        deal_card(self, card: Card.Card): Deals a card to the player.
        reward(self, amount: int): Rewards the player with chips.
        bet(self, amount: int) -> str: Places a bet by the player.
        get_action(self, window, high_bet: int, pot: int, table: list, players: list, blind: int): Gets the action chosen by the player.
        get_possible_actions(self, high_bet: int, blind: int) -> list: Returns the possible actions for the player.
        reset_cards_and_bet(self): Resets the cards and current bet of the player.
        get_AI_Resolver_action(self, high_bet: int, pot: int, table: list, players: list, blind: int): Gets the action chosen by the AI_resolve player.
        get_AI_Rollout_action(self, high_bet: int, pot: int, table: list, players: list, blind: int): Gets the action chosen by the AI player using rollout strategy.
    """

    def __init__(self, name: str, type: str = "human", index: int = 0):
        """
        Initializes a Player object.

        Args:
            name (str): The name of the player.
            type (str, optional): The type of the player. Defaults to "human".
            index (int, optional): The index of the player. Defaults to 0.
        """
        self.name = name
        self.type = type
        self.index = index
        self.chips = 0
        self.current_bet = 0
        self.cards = []
        self.active_in_hand = True
        self.is_all_in = False
        self.has_raised = False
        if type == "AI_resolve":
            self.action = None
            self.resolver = res.Resolver()
            self.player_range, self.opponent_range = util.generate_ranges()
    
    def __repr__(self):
        """
        Returns a string representation of the Player object.

        Returns:
            str: The string representation of the Player object.
        """
        return f"Player: {self.name} C: {self.chips}"

    def get_cards(self) -> list:
        """
        Returns the cards held by the player.

        Returns:
            list: The cards held by the player.
        """
        return self.cards

    def deal_card(self, card: Card.Card):
        """
        Deals a card to the player.

        Args:
            card (Card.Card): The card to be dealt.
        
        Raises:
            ValueError: If the player already has 2 cards.
        """
        if len(self.cards) < 2:
            self.cards.append(card)
        else:
            raise ValueError("Player can only have 2 cards")
    
    def reward(self, amount: int):
        """
        Rewards the player with chips.

        Args:
            amount (int): The amount of chips to be rewarded.
        """
        self.chips += amount

    def bet(self, amount: int) -> str:
        """
        Places a bet by the player.

        Args:
            amount (int): The amount of chips to be bet.

        Returns:
            str: The message indicating the bet made by the player.
        
        Raises:
            ValueError: If the amount is less than 0 or the player does not have any chips.
        """
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
        """
        Gets the action chosen by the player.

        Args:
            window: The window object for visualization.
            high_bet (int): The highest bet made by any player.
            pot (int): The current pot size.
            table (list): The cards on the table.
            players (list): The list of players in the game.
            blind (int): The blind amount.

        Returns:
            str: The action chosen by the player.
        """
        if self.type == "human":
            return util.visualize_human(window, table, self.cards, self.name, self.chips, pot, self.current_bet, high_bet, actions=self.get_possible_actions(high_bet, blind))
        elif self.type == "AI_resolve":
            self.action = None
            args = [copy.deepcopy(high_bet), copy.deepcopy(pot), copy.deepcopy(table), players, copy.deepcopy(blind)]
            thread = threading.Thread(target=self.get_AI_Resolver_action, args=args)
            thread.start()
            counter = 0
            while(not self.action):
                time.sleep(1)
                counter += 1
                gui.update_turn(window, self, players, counter)
            thread.join()
            return self.action
        else:
            util.visualize_AI(window, table, self.name, self.chips, pot, self.current_bet, high_bet)
            return self.get_AI_Rollout_action(high_bet, pot, table, players, blind)

        return "fold"
    
    def get_possible_actions(self, high_bet: int, blind: int) -> list:
        """
        Returns the possible actions for the player.

        Args:
            high_bet (int): The highest bet made by any player.
            blind (int): The blind amount.

        Returns:
            list: The list of possible actions for the player.
        """
        allowed_actions = config.get_actions()
        actions = ["fold"]
        if high_bet - self.current_bet >= 0 and self.chips >= high_bet - self.current_bet and "call" in allowed_actions:
            actions.append("call")
        if self.chips >= high_bet - self.current_bet + blind * 2 and self.has_raised == False and "bet" in allowed_actions:
            actions.append("bet")
        if self.chips > 0 and "all-in" in allowed_actions:
            actions.append("all-in")
        return actions

    def reset_cards_and_bet(self):
        """
        Resets the cards and current bet of the player.
        """
        self.cards = []
        self.current_bet = 0
        self.active_in_hand = True
        self.is_all_in = False
        self.has_raised = False

    def get_AI_Resolver_action(self, high_bet: int, pot: int, table: list, players: list, blind: int):
        """
        Gets the action chosen by the AI_resolve player.

        Args:
            high_bet (int): The highest bet made by any player.
            pot (int): The current pot size.
            table (list): The cards on the table.
            players (list): The list of players in the game.
            blind (int): The blind amount.
        """
        state_type = "decision"
        bets = {}
        player_stacks = {}
        has_raised = {}
        has_called = {}
        for player in players:
            bets[player.name] = copy.deepcopy(player.current_bet)
            player_stacks[player.name] = copy.deepcopy(player.chips)
            has_raised[player.name] = copy.deepcopy(player.has_raised)
            has_called[player.name] = False
        
        state = node.State("decision", bets, blind, player_stacks, table, self.name, has_raised, has_called)
        self.action = self.resolver.get_action(self, state)
        return

    def get_AI_Rollout_action(self, high_bet: int, pot: int, table: list, players: list, blind: int):
        """
        Gets the action chosen by the AI player using rollout strategy.

        Args:
            high_bet (int): The highest bet made by any player.
            pot (int): The current pot size.
            table (list): The cards on the table.
            players (list): The list of players in the game.
            blind (int): The blind amount.

        Returns:
            str: The action chosen by the AI player.
        """
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
        


