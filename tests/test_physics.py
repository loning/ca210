import random

from src.core import Rule110, make_leaf, make_node, make_ring


def lattice_bits(cell):
    return [leaf.x_prev for leaf in cell.children], [leaf.x_cur for leaf in cell.children]


def build_ring(x_prev_bits, x_cur_bits):
    leaves = [make_leaf(p, c) for p, c in zip(x_prev_bits, x_cur_bits)]
    return make_ring(leaves)


def snapshot_leaves(cell):
    data = []

    def _walk(node):
        if node.children:
            for child in node.children:
                _walk(child)
        else:
            data.append((node.x_prev, node.x_cur))

    _walk(cell)
    return data


def test_step_and_inverse_recover_state():
    leaves = [
        make_leaf(0, 1),
        make_leaf(1, 0),
        make_leaf(0, 0),
        make_leaf(1, 1),
    ]
    ring = make_ring(leaves)
    rule = Rule110()

    prev_before, cur_before = lattice_bits(ring)

    ring.step(rule)
    ring.step_inverse(rule)

    prev_after, cur_after = lattice_bits(ring)
    assert prev_before == prev_after
    assert cur_before == cur_after


def test_single_step_matches_manual_rule():
    leaves = [make_leaf(0, 0), make_leaf(0, 1), make_leaf(1, 1)]
    ring = make_ring(leaves)
    rule = Rule110()

    expected = []
    n = len(leaves)
    for idx, leaf in enumerate(leaves):
        left = leaves[(idx - 1) % n]
        right = leaves[(idx + 1) % n]
        next_bit = leaf.x_prev ^ rule.local(left.x_cur, leaf.x_cur, right.x_cur)
        expected.append(next_bit & 1)

    ring.step(rule)

    assert [leaf.x_cur for leaf in ring.children] == expected


def test_reversibility_multiple_random_steps():
    rng = random.Random(210)
    rule = Rule110()
    for size in (3, 5, 8, 13):
        for _ in range(10):
            x_prev = [rng.randint(0, 1) for _ in range(size)]
            x_cur = [rng.randint(0, 1) for _ in range(size)]
            ring = build_ring(x_prev, x_cur)
            init_prev, init_cur = lattice_bits(ring)
            steps = 7
            for _ in range(steps):
                ring.step(rule)
            for _ in range(steps):
                ring.step_inverse(rule)
            final_prev, final_cur = lattice_bits(ring)
            assert final_prev == init_prev
            assert final_cur == init_cur


def test_rule110_matches_naive_vector_step():
    rng = random.Random(358)
    size = 12
    x_prev = [rng.randint(0, 1) for _ in range(size)]
    x_cur = [rng.randint(0, 1) for _ in range(size)]
    rule = Rule110()

    ring = build_ring(x_prev, x_cur)
    ring.step(rule)
    ring_prev, ring_cur = lattice_bits(ring)

    expected_next = []
    for idx in range(size):
        l = x_cur[(idx - 1) % size]
        c = x_cur[idx]
        r = x_cur[(idx + 1) % size]
        next_bit = x_prev[idx] ^ rule.local(l, c, r)
        expected_next.append(next_bit & 1)
    naive_prev = x_cur[:]  # after step, previous becomes old current

    assert ring_prev == naive_prev
    assert ring_cur == expected_next


def test_rotation_equivariance():
    rule = Rule110()
    x_prev = [0, 1, 1, 0, 1, 0]
    x_cur = [1, 0, 1, 1, 0, 0]

    ring_a = build_ring(x_prev, x_cur)
    # Rotate arrays by k
    k = 2
    rot = lambda arr: arr[k:] + arr[:k]
    ring_b = build_ring(rot(x_prev), rot(x_cur))

    ring_a.step(rule)
    ring_b.step(rule)

    prev_a, cur_a = lattice_bits(ring_a)
    prev_b, cur_b = lattice_bits(ring_b)

    assert prev_b == rot(prev_a)
    assert cur_b == rot(cur_a)


def test_nested_ring_reversibility():
    rule = Rule110()

    inner1 = [make_leaf(0, 1), make_leaf(1, 0), make_leaf(0, 0)]
    inner2 = [make_leaf(1, 1), make_leaf(0, 1), make_leaf(1, 0)]

    node1 = make_node(0, 1, inner1)
    node2 = make_node(1, 0, inner2)
    node3 = make_node(0, 0, [make_leaf(1, 1), make_leaf(0, 0), make_leaf(1, 0)])

    ring = make_ring([node1, node2, node3])

    prev_before, cur_before = lattice_bits(ring)
    leaves_before = snapshot_leaves(ring)

    ring.step(rule)
    ring.step(rule)
    ring.step_inverse(rule)
    ring.step_inverse(rule)

    prev_after, cur_after = lattice_bits(ring)
    leaves_after = snapshot_leaves(ring)
    assert prev_before == prev_after
    assert cur_before == cur_after
    assert leaves_before == leaves_after
