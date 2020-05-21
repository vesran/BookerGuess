from collections import Counter
import numpy as np


def _frequency(data):
    counter = Counter()
    for one_set in data:
        for item in one_set:
            counter[item] += 1
    return counter


class FPTree:

    def __init__(self):
        self.item_nodes = None
        self.sets = None
        self.root = None
        self.frequencies = None

    def build(self, data, threshold):
        # Count item frequency
        self.frequencies = _frequency(data)

        # Sort items in sets
        self.item_nodes = {item: [] for item in self.frequencies}
        self.root = FPNode()
        most_frequent = {item for item in self.frequencies if self.frequencies[item] >= threshold}
        self.sets = []
        for items in data:
            self.sets.append(sorted(items.intersection(most_frequent), key=lambda x: self.frequencies[x], reverse=True))

        # Construct tree
        for items in self.sets:
            # Add path
            current = self.root
            for item in items:
                next = current.find_and_inc(item)
                if next is None:
                    next = current.add_and_return(item)
                    self.item_nodes[item].append(next)
                    next.parent = current
                current = next
        print("Done")

    def most_frequent(self, threshold):
        frequencies = Counter()
        for item in self.item_nodes:
            # Get all paths from the current item
            start_paths = self.item_nodes[item]
            paths = []
            for start in start_paths:
                # Read a path from leaf to root
                path = []
                current = start
                while current != self.root:
                    path.append(current)
                    frequencies[current] = current.occ
                    current = current.parent
            paths.append({})


class FPNode:

    def __init__(self, item=None):
        self.parent = None
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
    tree = FPTree()
    tree.build(data, 3)
