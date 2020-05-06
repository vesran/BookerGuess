from collections import Counter
import numpy as np


def _frequency(data):
    counter = Counter()
    for one_set in data:
        for item in one_set:
            counter[item] += 1
    return counter


class FPTree:

    def __init__(self, data):
        """
        :param data: List of all sets
        """
        # Count item frequency
        frequencies = _frequency(data)

        # Sort items in sets
        self.items = {item: FPNode(item=item) for item in frequencies}
        self.sets = [sorted(one_set, key=lambda x: frequencies[x], reverse=True) for one_set in data]  # ??!
        self.root = FPNode()


class FPNode:

    def __init__(self, item=None):
        self.item = item


if __name__ == '__main__':
    data = [
        {'a', 'b', 'c'},
        {'a', 'b'},
        {'a', 'c'},
        {'a', 'd'},
        {'a', 'c', 'd'},
    ]
    tree = FPTree(data)
