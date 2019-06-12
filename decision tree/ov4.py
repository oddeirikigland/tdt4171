import numpy as np
import os
import random

random_importance = False


class Tree:
    def __init__(self, test):
        self.test_attribute = test
        self.subtree = []

    def add_subtree(self, tree):
        self.subtree.append(tree)


def plurality_value(parent_examples):
    count_1 = 0
    count_2 = 0
    for example in parent_examples:
        if example[7] == 1.0:
            count_1 += 1
        else:
            count_2 += 1
    if count_1 == count_2:
        return random.choice((1.0, 2.0))
    return 1.0 if count_1 > count_2 else 2.0


def calculate_entropy(q):
    if q == 1.0 or q == 0.0:
        return 1 - q
    return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))


def calculate_remainder(pk, nk, p_and_n):
    return ((pk + nk) / p_and_n) * calculate_entropy(pk / (pk + nk))


def importance(attributes, examples):
    if random_importance:
        return attributes.pop(random.randrange(len(attributes)))
    else:
        count_1 = 0
        for example in examples:
            if example[7] == 1.0:
                count_1 += 1
        q = count_1 / len(examples)
        gain = {}

        entropy = calculate_entropy(q)
        for a in attributes:
            remainder = 0
            for value in [1, 2]:
                pk = 0
                nk = 0
                for example in examples:
                    if example[a - 1] == value:
                        pk += 1
                    else:
                        nk += 1
                remainder += calculate_remainder(pk, nk, len(examples))
            gain[a] = entropy - remainder
        attr = list(gain.values())
        best_attribute = attributes[attr.index(max(attr))]
        return best_attribute


def same_classification(examples):
    classification = examples[0][7]
    for example in examples[1:]:
        if example[7] != classification:
            return False
    return True


def decision_tree_learning(examples, attributes, parent_examples):
    if len(examples) == 0:
        return plurality_value(parent_examples)
    elif same_classification(examples):
        return examples[0][7]
    elif len(attributes) == 0:
        return plurality_value(examples)
    else:
        a = importance(attributes, examples)
        tree = Tree(a)
        for value in [1, 2]:
            eks = [example for example in examples if example[a - 1] == value]
            next_attributes = [attr for attr in attributes if attr != a]
            subtree = decision_tree_learning(eks, next_attributes, examples)
            tree.add_subtree(subtree)
        return tree


def run_example_on_tree(tree, example):
    if example[tree.test_attribute - 1] == 1:
        subtree = tree.subtree[0]
        if type(subtree) is not Tree:
            return subtree
        return run_example_on_tree(subtree, example)
    else:
        subtree = tree.subtree[1]
        if type(subtree) is not Tree:
            return subtree
        return run_example_on_tree(subtree, example)


def print_tree(tree, spacing=""):
    if type(tree) is not Tree:
        print(spacing + "Predicts: ", tree)
        return
    print(spacing + "Attribute: " + str(tree.test_attribute))

    print(spacing + " --> Value 1")
    print_tree(tree.subtree[0], spacing + "\t")

    print(spacing + " --> Value 2")
    print_tree(tree.subtree[1], spacing + "\t")


def get_file_from_path(rel_path):
    return np.loadtxt(os.path.join(os.path.dirname(__file__), rel_path))


def main():
    training_data_path = "data/training.txt"
    test_data_path = "data/test.txt"
    training_data = get_file_from_path(training_data_path)
    attributes = [1, 2, 3, 4, 5, 6, 7]
    tree = decision_tree_learning(training_data, attributes, [])

    test_data = get_file_from_path(test_data_path)
    score = 0
    for example in test_data:
        res = run_example_on_tree(tree, example)
        if res == example[7]:
            score += 1
    accuracy = score / len(test_data)
    print("Accuracy: " + str(accuracy))
    print_tree(tree)


if __name__ == "__main__":
    main()
