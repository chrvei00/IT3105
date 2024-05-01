import Util.Card as Card
class Range:
    def __init__(self):
        self.range_probability_distribution = self.gen_hole_pairs()
    
    def gen_hole_pairs(self) -> dict:
        hole_cards = {}
        cards = Card.Card.get_all_cards()  # Assuming you have a method to get all possible cards
        
        for card1 in cards:
            for card2 in cards:
                if card1 != card2:
                    hand = (card1, card2)
                    hole_cards[hand] = 0.0
        
        return hole_cards

    def update_range(self, hand: tuple, probability: float):
        self.range_probability_distribution[hand] = probability

    def get_range(self) -> dict:
        return self.range_probability_distribution

    def equalize_probabilities(self):
        for hand in self.range_probability_distribution:
            self.range_probability_distribution[hand] = 1 / len(self.range_probability_distribution)