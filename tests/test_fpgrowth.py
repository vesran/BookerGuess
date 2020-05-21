from tools.fpgrowth import FPNode, FPTree


def test_build_tree():
    data = [
        {'a', 'b', 'c'},
        {'a', 'b'},
        {'a', 'c'},
        {'a', 'd'},
        {'a', 'c', 'd'},
    ]
    tree = FPTree()
    tree.build(data, 1)

    assert tree.root.nexts[0].occ == 5
    assert tree.root.nexts[0].nexts[0].nexts.__len__() == 2
    assert tree.root.nexts[0].item == 'a'
    assert tree.root.nexts[0].nexts[0].item == 'c'
    assert tree.root.nexts[0].nexts[1].occ == 1
    assert tree.item_nodes['b'].__len__() == 2
    assert tree.root.nexts[0].parent == tree.root


def test_tree_threshold():
    data = [
        {'a', 'b', 'c'},
        {'a', 'b'},
        {'a', 'c'},
        {'a', 'd'},
        {'a', 'c', 'd'},
    ]
    tree = FPTree()
    tree.build(data, 3)

    assert tree.root.nexts.__len__() == 1
    assert tree.root.nexts[0].nexts.__len__() == 1
    assert tree.root.nexts[0].nexts[0].nexts.__len__() == 0
