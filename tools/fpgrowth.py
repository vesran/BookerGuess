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
        self.sets = [sorted(one_set, key=lambda x: frequencies[x], reverse=True) for one_set in data]
        self.root = FPNode()

    def build(self):
        lastest_nodes = {item: [] for item in self.items}
        for items in self.sets:
            # Add path
            current = self.root
            for item in items:
                next = current.find_and_inc(item)
                if next is None:
                    next = current.add_and_return(item)
                    lastest_nodes[item].append(next)
                current = next
        print("Done")


class FPNode:

    def __init__(self, item=None):
        self.item = item
        self.occ = 1
        self.nexts = []

    def find_and_inc(self, item):
        if len(self.nexts) > 0:
            for current in self.nexts:
                if current.item == item:
                    current.occ += 1
                    return current
        return None

    def add_and_return(self, item):
        another = FPNode(item)
        self.nexts.append(another)
        return another


if __name__ == '__main__':
    data = [
        {'a', 'b', 'c'},
        {'a', 'b'},
        {'a', 'c'},
        {'a', 'd'},
        {'a', 'c', 'd'},
    ]
    tree = FPTree(data)
    tree.build()
