import Util.Node as Node
import Util.State_Util as util

def generate_root(state: Node.State) -> Node.Node:
    """
    Generate the root state for the search tree.
    """
    return Node.Node(state=state, depth=0, regret_sum=util.gen_hole_pair_matrix(), strategy_sum=util.gen_hole_pair_matrix(), player_value=util.gen_range(), opponent_value=util.gen_range())

def generate_child_state(state: Node.State, object: str) -> Node.State:
    """
    Generate a child state depending on the action taken.
    """
    return util.gen_state(state=state, object=object)

def generate_children(node: Node.Node, end_depth: int, rollouts: int):
    """
    Generate all child states for the search tree.
    """
    depth = node.depth
    if depth < end_depth and node.children == []:
        if node.state.type == "decision":
            # print("Decision node, creating children for actions: ", util.possible_actions(node))
            for action in util.possible_actions(node):
                node.add_child( Node.Node(generate_child_state(state=node.state, object=action), regret_sum=util.gen_hole_pair_matrix(), strategy_sum=util.gen_hole_pair_matrix(), player_value=util.gen_range(), opponent_value=util.gen_range()) )
        elif node.state.type == "chance":
            # print("Chance node, creating children for cards: ", util.possible_cards(node.state, max=rollouts))
            for card in util.possible_cards(node.state, max=rollouts):
                node.add_child( Node.Node(generate_child_state(state=node.state, object=card), regret_sum=util.gen_hole_pair_matrix(), strategy_sum=util.gen_hole_pair_matrix(), player_value=util.gen_range(), opponent_value=util.gen_range()) )
        elif node.state.type == "terminal":
            # print("Terminal node, no children to create")
            return
    for child in node.children:
        generate_children(node=child, end_depth=end_depth, rollouts=rollouts)

def subtree_generator(state: Node.State, end_stage: str, end_depth: int, rollouts: int) -> Node:
    """
    Generate the subtree for the search tree.
    """
    root = generate_root(state=state)
    generate_children(root, end_depth, rollouts)
    return root