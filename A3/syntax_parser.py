import json


class TreeNode:
    """
    Class for representing parse trees. Contains some useful utilities for printing and reading/writing to string.
    """

    def from_list(lst):
        root = TreeNode(lst[0])
        root.children = [TreeNode.from_list(ls) for ls in lst[1:]]
        return root

    def from_string(string):
        return TreeNode.from_list(json.loads(string))

    def __init__(self, val):
        self.val = val
        self.children = []

    def to_list(self):
        return [self.val] + [c.to_list() for c in self.children]

    def to_string(self):
        return json.dumps(self.to_list())

    def display(self):
        string = self.val + '\n'
        stack = self.children
        done = False
        while not done:
            done = True
            new_stack = []
            for c in stack:
                string += c.val + '\t'
                if len(c.children) == 0:
                    new_stack.append(TreeNode('\t'))
                else:
                    done = False
                    new_stack.extend(c.children)
            string += '\n'
            stack = new_stack
        return string


with open('train_x.txt', 'r') as f:
    x = [l.split() for l in f.readlines()]
    # Each element of x is a list of words (a sentence).

with open('train_y.txt', 'r') as f:
    y = [TreeNode.from_string(l) for l in f.readlines()]
    # Each element of y is a TreeNode object representing the syntax of the corresponding element of x


# TODO estimate the PCFG that generated (x, y) and print to output. Your output should be a list of rules along with
# their corresponding probabilities (e.g. [(A -> B, 0.9), (A -> C, 0.1), ...]
def tree_preOder(root, preOrder):
    preOrder.append(root)
    if len(root.children) == 2:
        tree_preOder(root.children[0], preOrder)
        tree_preOder(root.children[1], preOrder)
    elif len(root.children) == 1:
        tree_preOder(root.children[0], preOrder)
    return preOrder


transition_count_dic = {}
symbols_count_dic = {}
probability_of_transition = {}

for treeRootNode in y:
    treeNodes = tree_preOder(treeRootNode, [])
    for eachNode in treeNodes:
        if len(eachNode.children) != 0:
            if eachNode.val in symbols_count_dic:
                symbols_count_dic[eachNode.val] += 1
            else:
                symbols_count_dic[eachNode.val] = 1
            # print(symbols_count_dic)
            s = "".join(str(x.val) for x in eachNode.children)
            if (eachNode.val, s) in transition_count_dic:
                transition_count_dic[(eachNode.val, s)] += 1
            else:
                transition_count_dic[(eachNode.val, s)] = 1

for transition in transition_count_dic.keys():
    probability_of_transition[str(transition[0]) + '->' + str(transition[1])] = transition_count_dic[transition] / \
                                                                                symbols_count_dic[transition[0]]

# print(symbols_count_dic)
# print(transition_count_dic)
# print(probability_of_transition)

PCFG_list = []

for key in sorted(probability_of_transition.keys()):
    # print(str(key) + ': ' + str(probability_of_transition[key]))  # separate line format of answer
    PCFG_list.append(str(key) + ': ' + str(probability_of_transition[key]))

print(PCFG_list)  # list format of answer
