from Game_Manager import Hand

def validate_game(Num_Human_Players, Num_AI_Players, Game_Type):
    if Num_Human_Players < 0 or Num_AI_Players < 0:
        raise ValueError("Number of players must be a non-negative integer")
    elif Num_Human_Players + Num_AI_Players < 2:
        raise ValueError("There must be at least 2 players")
    elif Num_Human_Players + Num_AI_Players > 6:
        raise ValueError("There can be at most 10 players")
    elif Game_Type not in ["simple", "complex"]:
        raise ValueError("Game type must be either simple or complex")

def validate_hand(hand: Hand):
    if hand is None:
        raise ValueError("Hand cannot be None")
    if hand.dealer is None:
        raise ValueError("Dealer cannot be None")
    if hand.players is None:
        raise ValueError("Players cannot be None")
    if hand.deck is None:
        raise ValueError("Deck cannot be None")
    if hand.pot is None:
        raise ValueError("Pot cannot be None")
    if hand.table is None:
        raise ValueError("Table cannot be None")

def setup_game(Num_Human_Players, Num_AI_Players, Game_Type):
    players = []
    # Create human players
    for i in range(Num_Human_Players):
        players.append(Player("human"))
    # Create AI players
    for i in range(Num_AI_Players):
        players.append(Player("AI"))
    # Randomly select a dealer
    dealer = random.choice(players)
    # Shuffle the players
    random.shuffle(players)
    # Create a deck
    deck = Deck()
    return players, dealer, deck

def rotate(players: list, dealer: Player):
    while players[-1] != dealer:
        players.append(players.pop(0))
    return players

def next_dealer(players: list, dealer: Player):
    dealer_index = players.index(dealer)
    if dealer_index == len(players) - 1:
        return players[0]
    else:
        return players[dealer_index + 1]

def hand_over(hand: Hand):
    if len(hand.players) == 1:
        return True
    elif len(hand.players) == 2:
        if hand.players[0].bet == hand.players[1].bet:
            return True
    return False

def round_over(hand: Hand):
    for player in hand.players:
        if player.bet != hand.pot/len(hand.players) and player.bet != 0:
            return False