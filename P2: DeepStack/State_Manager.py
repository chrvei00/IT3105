import Util.Node as Node
import Util.State_Util as util

def generate_root(state: Node.State) -> Node.Node:
    """
    Generate the root state for the search tree.
    """
    return Node.Node(state=state, depth=0)

def generate_child_state(state: Node.State, object: str) -> Node.State:
    """
    Generate a child state depending on the action taken.
    """
    return util.gen_state(state=state, object=object)

def generate_children(node: Node.Node, end_depth: int = 10):
    """
    Generate all child states for the search tree.
    """
    depth = node.depth
    while depth + 1 < end_depth and node.children == []:
        if node.state.type == "decision":
            for action in util.possible_actions(node):
                node.add_child( Node.Node(generate_child_state(state=node.state, object=action)) )
        elif node.state.type == "chance":
            for card in util.possible_cards(node.state):
                node.add_child( Node.Node(generate_child_state(state=node.state, object=card)) )
        elif node.state.type == "terminal":
            return
    for child in node.children:
        generate_children(child, end_depth=end_depth)

def subtree_generator(state: Node.State, end_stage: str, end_depth: int) -> Node:
    """
    Generate the subtree for the search tree.
    """
    root = generate_root(state=state)
    generate_children(root, end_depth)
    return root