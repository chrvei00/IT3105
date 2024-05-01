import Util.Node as Node
import Util.State_Util as util

def generate_root(state: Node.State) -> Node.Node:
    """
    Generate the root state for the search tree.

    Parameters:
    state (Node.State): The initial state of the search tree.

    Returns:
    Node.Node: The root node of the search tree.
    """
    return Node.Node(state=state, depth=0, regret_sum=util.gen_hole_pair_matrix(), strategy_sum=util.gen_hole_pair_matrix(), player_value=util.gen_range(), opponent_value=util.gen_range())

def generate_child_state(state: Node.State, object: str) -> Node.State:
    """
    Generate a child state depending on the action taken.

    Parameters:
    state (Node.State): The parent state.
    object (str): The action taken.

    Returns:
    Node.State: The child state.
    """
    return util.gen_state(state=state, object=object)

def generate_children(node: Node.Node, end_depth: int, rollouts: int):
    """
    Generate all child states for the search tree.

    Parameters:
    node (Node.Node): The current node.
    end_depth (int): The maximum depth of the search tree.
    rollouts (int): The number of rollouts to perform.

    Returns:
    None
    """
    depth = node.depth
    if depth < end_depth and node.children == []:
        if node.state.type == "decision":
            for action in util.possible_actions(node):
                node.add_child( Node.Node(generate_child_state(state=node.state, object=action), action=action, regret_sum=util.gen_hole_pair_matrix(), strategy_sum=util.gen_hole_pair_matrix(), player_value=util.gen_range(), opponent_value=util.gen_range()) )
        elif node.state.type == "chance":
            for card in util.possible_cards(node.state, max=rollouts):
                node.add_child( Node.Node(generate_child_state(state=node.state, object=card), regret_sum=util.gen_hole_pair_matrix(), strategy_sum=util.gen_hole_pair_matrix(), player_value=util.gen_range(), opponent_value=util.gen_range()) )
        elif node.state.type == "terminal":
            return
    for child in node.children:
        generate_children(node=child, end_depth=end_depth, rollouts=rollouts)

def subtree_generator(state: Node.State, end_stage: str, end_depth: int, rollouts: int) -> Node:
    """
    Generate the subtree for the search tree.

    Parameters:
    state (Node.State): The initial state of the search tree.
    end_stage (str): The stage at which to stop generating child states.
    end_depth (int): The maximum depth of the search tree.
    rollouts (int): The number of rollouts to perform.

    Returns:
    Node: The root node of the generated subtree.
    """
    root = generate_root(state=state)
    generate_children(root, end_depth, rollouts)
    return root