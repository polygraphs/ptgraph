"""
Test suite for pure-PyTorch DGL replacement (ptgraph).

Usage:
    # Full tests including comparison against real DGL:
    python test_dgl.py

    # Standalone mode (no dgl needed, tests internal consistency only):
    python test_dgl.py --standalone

Requires: torch, networkx, pytest, ptgraph
Optional: dgl (for comparison tests)
"""

import pytest
import torch
import networkx as nx
import tempfile
import os
import sys

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import ptgraph

# Try importing real DGL for comparison
try:
    import dgl as _real_dgl
    HAS_DGL = True
except ImportError:
    _real_dgl = None
    HAS_DGL = False

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def make_triangle():
    """0→1, 1→2, 2→0"""
    return [0, 1, 2], [1, 2, 0]


def make_triangle_with_self_loop():
    """0→1, 1→2, 2→0, 0→0"""
    return [0, 1, 2, 0], [1, 2, 0, 0]


def make_star():
    """0→1, 0→2, 0→3, 0→4"""
    return [0, 0, 0, 0], [1, 2, 3, 4]


def make_disconnected():
    """0→1, 2→3 (node 4 isolated)"""
    return [0, 2], [1, 3]


def allclose(a, b, atol=1e-6):
    return torch.allclose(a.float(), b.float(), atol=atol)


# ──────────────────────────────────────────────────────────────────────
# 1. Graph Construction
# ──────────────────────────────────────────────────────────────────────

class TestGraphConstruction:
    def test_basic(self):
        src, dst = make_triangle()
        g = ptgraph.graph((src, dst))
        assert g.num_nodes() == 3
        assert g.num_edges() == 3

    def test_num_nodes_explicit(self):
        g = ptgraph.graph(([0], [1]), num_nodes=10)
        assert g.num_nodes() == 10
        assert g.num_edges() == 1

    def test_empty_graph(self):
        g = ptgraph.graph(([], []), num_nodes=5)
        assert g.num_nodes() == 5
        assert g.num_edges() == 0

    def test_tensor_input(self):
        src = torch.tensor([0, 1, 2])
        dst = torch.tensor([1, 2, 0])
        g = ptgraph.graph((src, dst))
        assert g.num_nodes() == 3

    def test_ndata_edata(self):
        g = ptgraph.graph(make_triangle())
        g.ndata['h'] = torch.randn(3, 4)
        g.edata['w'] = torch.randn(3)
        assert g.ndata['h'].shape == (3, 4)
        assert g.edata['w'].shape == (3,)

    def test_edges_uv(self):
        src, dst = make_triangle()
        g = ptgraph.graph((src, dst))
        s, d = g.edges()
        assert allclose(s, torch.tensor(src))
        assert allclose(d, torch.tensor(dst))

    def test_edges_all(self):
        g = ptgraph.graph(make_triangle())
        s, d, eid = g.edges(form='all')
        assert eid.shape[0] == 3
        assert allclose(eid, torch.arange(3))

    def test_nodes(self):
        g = ptgraph.graph(make_triangle())
        assert allclose(g.nodes(), torch.arange(3))

    def test_number_of_aliases(self):
        g = ptgraph.graph(make_triangle())
        assert g.number_of_nodes() == g.num_nodes()
        assert g.number_of_edges() == g.num_edges()


# ──────────────────────────────────────────────────────────────────────
# 2. Degrees
# ──────────────────────────────────────────────────────────────────────

class TestDegrees:
    def test_in_degrees_triangle(self):
        g = ptgraph.graph(make_triangle())
        deg = g.in_degrees()
        assert allclose(deg, torch.ones(3, dtype=torch.long))

    def test_out_degrees_star(self):
        g = ptgraph.graph(make_star())
        deg = g.out_degrees()
        assert deg[0] == 4
        assert (deg[1:] == 0).all()

    def test_in_degrees_star(self):
        g = ptgraph.graph(make_star())
        deg = g.in_degrees()
        assert deg[0] == 0
        assert (deg[1:] == 1).all()

    def test_in_degrees_subset(self):
        g = ptgraph.graph(make_star())
        deg = g.in_degrees(torch.tensor([1, 3]))
        assert allclose(deg, torch.ones(2, dtype=torch.long))

    def test_disconnected(self):
        g = ptgraph.graph(make_disconnected(), num_nodes=5)
        assert g.in_degrees()[4] == 0
        assert g.out_degrees()[4] == 0


# ──────────────────────────────────────────────────────────────────────
# 3. remove_self_loop
# ──────────────────────────────────────────────────────────────────────

class TestRemoveSelfLoop:
    def test_basic(self):
        g = ptgraph.graph(make_triangle_with_self_loop())
        g.edata['w'] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        g2 = ptgraph.remove_self_loop(g)
        assert g2.num_edges() == 3
        assert allclose(g2.edata['w'], torch.tensor([1.0, 2.0, 3.0]))

    def test_no_self_loops(self):
        g = ptgraph.graph(make_triangle())
        g2 = ptgraph.remove_self_loop(g)
        assert g2.num_edges() == 3

    def test_preserves_ndata(self):
        g = ptgraph.graph(make_triangle_with_self_loop())
        g.ndata['h'] = torch.randn(3, 4)
        g2 = ptgraph.remove_self_loop(g)
        assert allclose(g2.ndata['h'], g.ndata['h'])


# ──────────────────────────────────────────────────────────────────────
# 4. add_self_loop (via transforms)
# ──────────────────────────────────────────────────────────────────────

class TestAddSelfLoop:
    def test_basic(self):
        g = ptgraph.graph(make_triangle())
        g2 = ptgraph.transforms.add_self_loop(g)
        assert g2.num_edges() == 6  # 3 original + 3 self-loops

    def test_edata_padded(self):
        g = ptgraph.graph(make_triangle())
        g.edata['w'] = torch.ones(3)
        g2 = ptgraph.transforms.add_self_loop(g)
        assert g2.edata['w'].shape[0] == 6
        # New edges get zero-padded
        assert allclose(g2.edata['w'][3:], torch.zeros(3))


# ──────────────────────────────────────────────────────────────────────
# 5. filter_edges
# ──────────────────────────────────────────────────────────────────────

class TestFilterEdges:
    def test_src_lt_dst(self):
        g = ptgraph.graph(([0, 1, 2, 3], [1, 0, 3, 2]))
        kept = g.filter_edges(lambda e: e.src_ids() < e.dst_ids())
        assert allclose(kept, torch.tensor([0, 2]))

    def test_using_ndata(self):
        g = ptgraph.graph(([0, 1, 2], [1, 2, 0]))
        g.ndata['val'] = torch.tensor([10.0, 20.0, 30.0])
        kept = g.filter_edges(lambda e: e.src['val'] > 15.0)
        assert allclose(kept, torch.tensor([1, 2]))

    def test_all_filtered(self):
        g = ptgraph.graph(make_triangle())
        kept = g.filter_edges(lambda e: torch.zeros(3, dtype=torch.bool))
        assert kept.shape[0] == 0

    def test_len_edges(self):
        g = ptgraph.graph(make_triangle())
        def fn(edges):
            return torch.ones(len(edges), dtype=torch.bool)
        kept = g.filter_edges(fn)
        assert kept.shape[0] == 3


# ──────────────────────────────────────────────────────────────────────
# 6. send_and_recv — builtin reduce
# ──────────────────────────────────────────────────────────────────────

class TestSendRecvBuiltin:
    def _msg_copy_src(self, edges):
        return {'m': edges.src['h']}

    def test_sum_reduce(self):
        g = ptgraph.graph(make_triangle())
        g.ndata['h'] = torch.tensor([[1.0], [2.0], [3.0]])
        g.send_and_recv(
            torch.arange(3),
            self._msg_copy_src,
            'sum'
        )
        # 0←2 (3.0), 1←0 (1.0), 2←1 (2.0)
        expected = torch.tensor([[3.0], [1.0], [2.0]])
        assert allclose(g.ndata['m'], expected)

    def test_mean_reduce(self):
        # Star: 0→1, 0→2; 1→2
        g = ptgraph.graph(([0, 0, 1], [1, 2, 2]))
        g.ndata['h'] = torch.tensor([[10.0], [20.0], [30.0]])
        g.send_and_recv(
            torch.arange(3),
            self._msg_copy_src,
            'mean'
        )
        # node 0: no incoming → keep old value
        # node 1: receives from 0 → 10.0
        # node 2: receives from 0 (10.0) and 1 (20.0) → mean = 15.0
        assert g.ndata['m'][1].item() == pytest.approx(10.0)
        assert g.ndata['m'][2].item() == pytest.approx(15.0)

    def test_non_receiving_nodes_preserved(self):
        """Critical: nodes with no incoming messages keep their old values."""
        g = ptgraph.graph(([0], [1]), num_nodes=3)
        g.ndata['h'] = torch.tensor([[1.0], [2.0], [3.0]])
        g.ndata['m'] = torch.tensor([[99.0], [99.0], [99.0]])
        g.send_and_recv(
            torch.arange(1),
            self._msg_copy_src,
            'sum'
        )
        # Node 1 receives message (1.0), nodes 0 and 2 untouched
        assert g.ndata['m'][0].item() == pytest.approx(99.0)
        assert g.ndata['m'][1].item() == pytest.approx(1.0)
        assert g.ndata['m'][2].item() == pytest.approx(99.0)

    def test_with_edge_data(self):
        g = ptgraph.graph(make_triangle())
        g.ndata['h'] = torch.tensor([[1.0], [2.0], [3.0]])
        g.edata['w'] = torch.tensor([0.5, 1.0, 2.0])

        def msg_fn(edges):
            return {'m': edges.src['h'] * edges.data['w'].unsqueeze(-1)}

        g.send_and_recv(torch.arange(3), msg_fn, 'sum')
        # 0←2 (3.0*2.0=6.0), 1←0 (1.0*0.5=0.5), 2←1 (2.0*1.0=2.0)
        expected = torch.tensor([[6.0], [0.5], [2.0]])
        assert allclose(g.ndata['m'], expected)

    def test_subset_of_edges(self):
        g = ptgraph.graph(make_triangle())
        g.ndata['h'] = torch.tensor([[1.0], [2.0], [3.0]])
        # Only send along edge 0 (0→1)
        g.send_and_recv(torch.tensor([0]), self._msg_copy_src, 'sum')
        assert g.ndata['m'][1].item() == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────────────
# 7. send_and_recv — UDF reduce (degree bucketing)
# ──────────────────────────────────────────────────────────────────────

class TestSendRecvUDF:
    def test_sum_udf(self):
        g = ptgraph.graph(make_triangle())
        g.ndata['h'] = torch.tensor([[1.0], [2.0], [3.0]])
        g.send_and_recv(
            torch.arange(3),
            lambda e: {'m': e.src['h']},
            lambda n: {'m': torch.sum(n.mailbox['m'], dim=1)},
        )
        expected = torch.tensor([[3.0], [1.0], [2.0]])
        assert allclose(g.ndata['m'], expected)

    def test_mean_udf_no_padding(self):
        """
        Critical test: UDF mean must not be corrupted by zero-padding.

        Graph: 0→2, 1→2, 0→1
        Node 1 has in-degree 1, node 2 has in-degree 2.
        If mailbox is zero-padded to max_deg=2, mean for node 1 would
        be halved. Degree bucketing avoids this.
        """
        g = ptgraph.graph(([0, 1, 0], [2, 2, 1]))
        g.ndata['h'] = torch.tensor([[10.0], [20.0], [30.0]])
        g.send_and_recv(
            torch.arange(3),
            lambda e: {'m': e.src['h']},
            lambda n: {'m': torch.mean(n.mailbox['m'], dim=1)},
        )
        # Node 1: receives 10.0 → mean = 10.0 (NOT 5.0 from padding)
        # Node 2: receives 10.0, 20.0 → mean = 15.0
        assert g.ndata['m'][1].item() == pytest.approx(10.0)
        assert g.ndata['m'][2].item() == pytest.approx(15.0)

    def test_max_udf_no_padding(self):
        g = ptgraph.graph(([0, 1, 0], [2, 2, 1]))
        g.ndata['h'] = torch.tensor([[-5.0], [-3.0], [0.0]])
        g.send_and_recv(
            torch.arange(3),
            lambda e: {'m': e.src['h']},
            lambda n: {'m': torch.max(n.mailbox['m'], dim=1).values},
        )
        # Node 1: receives -5.0 → max = -5.0 (NOT 0.0 from padding)
        # Node 2: receives -5.0, -3.0 → max = -3.0
        assert g.ndata['m'][1].item() == pytest.approx(-5.0)
        assert g.ndata['m'][2].item() == pytest.approx(-3.0)

    def test_mailbox_shape(self):
        """Verify mailbox shape matches exact in-degree per bucket."""
        g = ptgraph.graph(([0, 1, 2, 0], [3, 3, 3, 1]))
        g.ndata['h'] = torch.ones(4, 2)
        shapes_seen = []

        def capture_reduce(nodes):
            for k, v in nodes.mailbox.items():
                shapes_seen.append(v.shape)
            return {'m': torch.sum(nodes.mailbox['m'], dim=1)}

        g.send_and_recv(
            torch.arange(4),
            lambda e: {'m': e.src['h']},
            capture_reduce,
        )
        # Node 1: deg=1 → (1, 1, 2); Node 3: deg=3 → (1, 3, 2)
        assert (1, 1, 2) in shapes_seen
        assert (1, 3, 2) in shapes_seen
        # No shape should have padding dim
        for s in shapes_seen:
            assert s[1] in (1, 3)  # only exact degrees


# ──────────────────────────────────────────────────────────────────────
# 8. send_and_recv — apply function
# ──────────────────────────────────────────────────────────────────────

class TestApplyFn:
    def test_apply_reads_ndata(self):
        g = ptgraph.graph(make_triangle())
        g.ndata['h'] = torch.tensor([[1.0], [2.0], [3.0]])
        g.ndata['bias'] = torch.tensor([[0.1], [0.2], [0.3]])

        def apply_fn(nodes):
            return {'out': nodes.data['m'] + nodes.data['bias']}

        g.send_and_recv(
            torch.arange(3),
            lambda e: {'m': e.src['h']},
            'sum',
            apply_fn,
        )
        # m: [3.0, 1.0, 2.0], bias: [0.1, 0.2, 0.3]
        expected = torch.tensor([[3.1], [1.2], [2.3]])
        assert allclose(g.ndata['out'], expected)

    def test_apply_only_on_recv_nodes(self):
        """Apply function should only run on receiving nodes."""
        g = ptgraph.graph(([0], [1]), num_nodes=3)
        g.ndata['h'] = torch.tensor([[1.0], [2.0], [3.0]])
        g.ndata['flag'] = torch.tensor([0.0, 0.0, 0.0])

        def apply_fn(nodes):
            return {'flag': torch.ones(nodes.data['flag'].shape[0])}

        g.send_and_recv(
            torch.arange(1),
            lambda e: {'m': e.src['h']},
            'sum',
            apply_fn,
        )
        # Only node 1 received → only node 1's flag is set
        assert g.ndata['flag'][0].item() == 0.0
        assert g.ndata['flag'][1].item() == 1.0
        assert g.ndata['flag'][2].item() == 0.0


# ──────────────────────────────────────────────────────────────────────
# 9. NetworkX conversion
# ──────────────────────────────────────────────────────────────────────

class TestNetworkX:
    def test_to_networkx(self):
        g = ptgraph.graph(make_triangle())
        G = ptgraph.to_networkx(g)
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 3

    def test_from_networkx_directed(self):
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        g = ptgraph.from_networkx(G)
        assert g.num_nodes() == 3
        assert g.num_edges() == 3

    def test_from_networkx_undirected(self):
        """DGL doubles edges for undirected graphs."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2)])
        g = ptgraph.from_networkx(G)
        assert g.num_nodes() == 3
        assert g.num_edges() == 4  # 2 edges × 2 directions

    def test_from_networkx_undirected_complete(self):
        G = nx.complete_graph(4)  # undirected, 6 edges
        g = ptgraph.from_networkx(G)
        assert g.num_edges() == 12  # 6 × 2

    def test_roundtrip(self):
        g = ptgraph.graph(make_triangle())
        g.ndata['h'] = torch.tensor([1.0, 2.0, 3.0])
        G = ptgraph.to_networkx(g, node_attrs=['h'])
        g2 = ptgraph.from_networkx(G, node_attrs=['h'])
        assert g2.num_nodes() == 3
        assert allclose(g2.ndata['h'], g.ndata['h'])

    def test_from_networkx_node_attrs(self):
        G = nx.DiGraph()
        G.add_node(0, val=1.0)
        G.add_node(1, val=2.0)
        G.add_edge(0, 1)
        g = ptgraph.from_networkx(G, node_attrs=['val'])
        assert allclose(g.ndata['val'], torch.tensor([1.0, 2.0]))


# ──────────────────────────────────────────────────────────────────────
# 10. Save / Load
# ──────────────────────────────────────────────────────────────────────

class TestSaveLoad:
    def test_roundtrip_single(self):
        g = ptgraph.graph(make_triangle())
        g.ndata['h'] = torch.randn(3, 4)
        g.edata['w'] = torch.randn(3)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.pt")
            ptgraph.save_graphs(path, g)
            loaded, labels = ptgraph.load_graphs(path)
            assert len(loaded) == 1
            assert loaded[0].num_edges() == 3
            assert allclose(loaded[0].ndata['h'], g.ndata['h'])
            assert allclose(loaded[0].edata['w'], g.edata['w'])

    def test_roundtrip_multiple(self):
        g1 = ptgraph.graph(make_triangle())
        g2 = ptgraph.graph(make_star())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.pt")
            ptgraph.save_graphs(path, [g1, g2], labels={"y": torch.tensor([0, 1])})
            loaded, labels = ptgraph.load_graphs(path)
            assert len(loaded) == 2
            assert loaded[0].num_edges() == 3
            assert loaded[1].num_edges() == 4
            assert allclose(labels['y'], torch.tensor([0, 1]))

    def test_load_idx_list(self):
        g1 = ptgraph.graph(make_triangle())
        g2 = ptgraph.graph(make_star())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.pt")
            ptgraph.save_graphs(path, [g1, g2])
            loaded, _ = ptgraph.load_graphs(path, idx_list=[1])
            assert len(loaded) == 1
            assert loaded[0].num_edges() == 4

    def test_bin_extension_works(self):
        """Files with .bin extension should work fine."""
        g = ptgraph.graph(make_triangle())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.bin")
            ptgraph.save_graphs(path, g)
            loaded, _ = ptgraph.load_graphs(path)
            assert loaded[0].num_edges() == 3


# ──────────────────────────────────────────────────────────────────────
# 11. random.seed
# ──────────────────────────────────────────────────────────────────────

class TestRandom:
    def test_seed(self):
        ptgraph.random.seed(42)
        a = torch.randn(5)
        ptgraph.random.seed(42)
        b = torch.randn(5)
        assert allclose(a, b)


# ──────────────────────────────────────────────────────────────────────
# 12. Bala-Goyal simulation smoke test
# ──────────────────────────────────────────────────────────────────────

class TestBalaGoyalSmoke:
    """
    Mimics the key operations from the PolyGraphOp / BalaGoyalOp code
    to ensure the full pipeline works end-to-end.
    """

    def test_full_pipeline(self):
        torch.manual_seed(42)

        # Build a small cycle: 0→1→2→3→0 (directed)
        g = ptgraph.graph(([0, 1, 2, 3], [1, 2, 3, 0]))
        n = g.num_nodes()

        # Init beliefs
        g.ndata['beliefs'] = torch.full((n,), 0.6)
        g.ndata['logits'] = torch.zeros(n)

        # Simulate experiment: payoffs = [successes, trials]
        g.ndata['payoffs'] = torch.tensor([
            [3.0, 5.0],
            [0.0, 0.0],   # node 1 doesn't experiment
            [4.0, 5.0],
            [2.0, 5.0],
        ])

        # Filter: only edges where source has trials > 0
        def filterfn(edges):
            return torch.gt(edges.src['payoffs'][:, 1], 0.0)

        # Message: send payoffs
        def messagefn(edges):
            return {'payoffs': edges.src['payoffs']}

        # Reduce: sum payoffs
        def reducefn(nodes):
            return {'payoffs': torch.sum(nodes.mailbox['payoffs'], dim=1)}

        # Apply: update beliefs
        def applyfn(nodes):
            return {'beliefs': nodes.data['beliefs'] * 0.9 + 0.1}

        edges = g.filter_edges(filterfn)
        g.send_and_recv(edges, messagefn, reducefn, applyfn)

        # Node 1 receives from node 0 (has payoffs)
        # Node 3 receives from node 2 (has payoffs)
        # Node 0 receives from node 3 (has payoffs)
        # Node 2 receives from node 1 (NO payoffs, filtered out)
        assert g.ndata['payoffs'][1][0].item() == pytest.approx(3.0)
        assert g.ndata['payoffs'][3][0].item() == pytest.approx(4.0)

        # Node 2 was NOT a recv node, so its beliefs should be unchanged
        assert g.ndata['beliefs'][2].item() == pytest.approx(0.6)
        # Other nodes were recv'd, so applyfn ran: 0.6 * 0.9 + 0.1 = 0.64
        assert g.ndata['beliefs'][1].item() == pytest.approx(0.64)

    def test_multiple_steps(self):
        """Run 5 steps to check no state corruption."""
        torch.manual_seed(123)
        g = ptgraph.graph(([0, 1, 2], [1, 2, 0]))
        n = g.num_nodes()
        g.ndata['beliefs'] = torch.full((n,), 0.5)
        g.ndata['logits'] = torch.zeros(n)

        for step in range(5):
            g.ndata['payoffs'] = torch.rand(n, 2)

            edges = g.filter_edges(
                lambda e: e.src['payoffs'][:, 1] > 0.3
            )
            if edges.shape[0] > 0:
                g.send_and_recv(
                    edges,
                    lambda e: {'payoffs': e.src['payoffs']},
                    lambda n: {'payoffs': torch.sum(n.mailbox['payoffs'], dim=1)},
                    lambda n: {'beliefs': torch.clamp(n.data['beliefs'] + 0.01, 0, 1)},
                )

        # Just check nothing crashed and beliefs are in valid range
        assert (g.ndata['beliefs'] >= 0).all()
        assert (g.ndata['beliefs'] <= 1).all()


# ──────────────────────────────────────────────────────────────────────
# 13. NoOp — empty message/reduce dicts
# ──────────────────────────────────────────────────────────────────────

class TestNoOp:
    """Tests the NoOp pattern where message/reduce return empty dicts."""

    def test_empty_message_reduce(self):
        g = ptgraph.graph(make_triangle())
        g.ndata['beliefs'] = torch.tensor([0.6, 0.7, 0.8])
        g.ndata['payoffs'] = torch.randn(3, 2)

        edges = torch.arange(g.num_edges())
        # Should not crash, and ndata should be unchanged
        g.send_and_recv(
            edges,
            lambda e: {},
            lambda n: {},
        )
        assert allclose(g.ndata['beliefs'], torch.tensor([0.6, 0.7, 0.8]))

    def test_empty_message_with_apply(self):
        g = ptgraph.graph(make_triangle())
        g.ndata['beliefs'] = torch.tensor([0.6, 0.7, 0.8])

        def applyfn(nodes):
            return nodes.data

        g.send_and_recv(
            torch.arange(g.num_edges()),
            lambda e: {},
            lambda n: {},
            applyfn,
        )
        assert allclose(g.ndata['beliefs'], torch.tensor([0.6, 0.7, 0.8]))

    def test_empty_with_filter(self):
        """All edges filtered out → zero edges passed to send_and_recv."""
        g = ptgraph.graph(make_triangle())
        g.ndata['beliefs'] = torch.tensor([0.6, 0.7, 0.8])

        edges = g.filter_edges(lambda e: torch.zeros(3, dtype=torch.bool))
        assert edges.shape[0] == 0
        # send_and_recv with zero edges should be a no-op
        if edges.shape[0] > 0:
            g.send_and_recv(edges, lambda e: {}, lambda n: {})
        assert allclose(g.ndata['beliefs'], torch.tensor([0.6, 0.7, 0.8]))


# ──────────────────────────────────────────────────────────────────────
# 14. OConnorWeatherall — multi-key mailbox, per-neighbor iteration
# ──────────────────────────────────────────────────────────────────────

class TestOConnorWeatherallPattern:
    """
    Tests the OConnorWeatherall reduce pattern:
    - Sends multiple keys (payoffs + beliefs)
    - Iterates through neighbors by index in the reduce
    - Uses len(nodes)
    """

    def test_multi_key_mailbox(self):
        """Message sends two keys; reduce receives both."""
        g = ptgraph.graph(([0, 1], [2, 2]))
        g.ndata['h'] = torch.tensor([[1.0], [2.0], [3.0]])
        g.ndata['b'] = torch.tensor([0.1, 0.2, 0.3])

        def msg_fn(edges):
            return {'h': edges.src['h'], 'b': edges.src['b']}

        def reduce_fn(nodes):
            assert 'h' in nodes.mailbox
            assert 'b' in nodes.mailbox
            h_sum = torch.sum(nodes.mailbox['h'], dim=1)
            b_sum = torch.sum(nodes.mailbox['b'], dim=1)
            return {'h': h_sum, 'b_agg': b_sum}

        g.send_and_recv(torch.arange(2), msg_fn, reduce_fn)
        assert g.ndata['h'][2].item() == pytest.approx(3.0)  # 1+2
        assert g.ndata['b_agg'][2].item() == pytest.approx(0.3)  # 0.1+0.2

    def test_per_neighbor_iteration(self):
        """
        Reduce iterates through neighbors by index, mimicking
        OConnorWeatherall's `for i in range(neighbours)` loop.
        """
        # 0→2, 1→2 (node 2 has 2 neighbors)
        g = ptgraph.graph(([0, 1], [2, 2]))
        g.ndata['val'] = torch.tensor([10.0, 20.0, 0.0])
        g.ndata['belief'] = torch.tensor([0.5, 0.8, 0.6])

        def msg_fn(edges):
            return {
                'val': edges.src['val'].unsqueeze(-1),
                'belief': edges.src['belief'],
            }

        def reduce_fn(nodes):
            _, neighbours = nodes.mailbox['belief'].shape
            prior = nodes.data['belief']
            for i in range(neighbours):
                val_i = nodes.mailbox['val'][:, i, 0]
                belief_i = nodes.mailbox['belief'][:, i]
                delta = torch.abs(prior - belief_i)
                # Simple update: prior moves toward neighbor
                prior = prior + val_i * 0.01 * (1.0 - delta)
            return {'belief': prior}

        g.send_and_recv(torch.arange(2), msg_fn, reduce_fn)
        # Just verify it runs and produces a modified belief for node 2
        assert g.ndata['belief'][2].item() != 0.6  # changed
        # Nodes 0, 1 not recv'd — unchanged
        assert g.ndata['belief'][0].item() == pytest.approx(0.5)
        assert g.ndata['belief'][1].item() == pytest.approx(0.8)

    def test_len_nodes_in_reduce(self):
        """len(nodes) must work inside reduce — used by OConnorWeatherall."""
        g = ptgraph.graph(([0, 1, 2], [3, 3, 3]))
        g.ndata['h'] = torch.ones(4, 2)

        len_seen = []

        def reduce_fn(nodes):
            len_seen.append(len(nodes))
            return {'h': torch.sum(nodes.mailbox['m'], dim=1)}

        g.send_and_recv(
            torch.arange(3),
            lambda e: {'m': e.src['h']},
            reduce_fn,
        )
        # Only node 3 receives, so len(nodes) should be 1
        assert 1 in len_seen

    def test_len_nodes_multiple_buckets(self):
        """len(nodes) with multiple degree buckets."""
        # 0→2 (deg 1), 0→3, 1→3 (deg 2)
        g = ptgraph.graph(([0, 0, 1], [2, 3, 3]))
        g.ndata['h'] = torch.ones(4, 1)

        lens_seen = []

        def reduce_fn(nodes):
            lens_seen.append(len(nodes))
            return {'m': torch.sum(nodes.mailbox['m'], dim=1)}

        g.send_and_recv(
            torch.arange(3),
            lambda e: {'m': e.src['h']},
            reduce_fn,
        )
        # Bucket deg=1: node 2 (len=1), bucket deg=2: node 3 (len=1)
        assert sorted(lens_seen) == [1, 1]

    def test_zeros_with_len_nodes(self):
        """torch.zeros((len(nodes),)) pattern from OConnorWeatherall."""
        g = ptgraph.graph(([0, 1], [2, 2]))
        g.ndata['h'] = torch.ones(3, 1)

        def reduce_fn(nodes):
            z = torch.zeros((len(nodes),))
            result = torch.sum(nodes.mailbox['m'], dim=1).squeeze(-1) + z
            return {'out': result}

        g.send_and_recv(
            torch.arange(2),
            lambda e: {'m': e.src['h']},
            reduce_fn,
        )
        assert g.ndata['out'][2].item() == pytest.approx(2.0)

    def test_varying_degrees_multi_key(self):
        """
        Full OConnorWeatherall-like scenario: nodes with different degrees,
        multi-key mailbox, per-neighbor iteration with len(nodes).
        """
        # Node 3 has deg 3 (from 0,1,2), node 4 has deg 1 (from 2)
        g = ptgraph.graph(([0, 1, 2, 2], [3, 3, 3, 4]))
        n = g.num_nodes()
        g.ndata['payoffs'] = torch.tensor([
            [3.0, 5.0], [2.0, 5.0], [4.0, 5.0], [0.0, 0.0], [0.0, 0.0],
        ])
        g.ndata['beliefs'] = torch.tensor([0.6, 0.7, 0.8, 0.5, 0.5])
        g.ndata['logits'] = torch.zeros(n)

        def msg_fn(edges):
            return {
                'payoffs': edges.src['payoffs'],
                'beliefs': edges.src['beliefs'],
            }

        def reduce_fn(nodes):
            _, neighbours = nodes.mailbox['beliefs'].shape
            prior = nodes.data['beliefs']
            for i in range(neighbours):
                values = nodes.mailbox['payoffs'][:, i, 0]
                trials = nodes.mailbox['payoffs'][:, i, 1]
                belief_i = nodes.mailbox['beliefs'][:, i]
                delta = torch.abs(prior - belief_i)
                # Simplified update
                update = values / trials.clamp(min=1) * 0.1 * (1.0 - delta)
                prior = prior + update
            return {'beliefs': prior}

        g.send_and_recv(torch.arange(4), msg_fn, reduce_fn)

        # Node 3 (deg 3) and node 4 (deg 1) should be updated
        assert g.ndata['beliefs'][3].item() != 0.5
        assert g.ndata['beliefs'][4].item() != 0.5
        # Nodes 0,1,2 not recv'd — unchanged
        assert g.ndata['beliefs'][0].item() == pytest.approx(0.6)
        assert g.ndata['beliefs'][1].item() == pytest.approx(0.7)
        assert g.ndata['beliefs'][2].item() == pytest.approx(0.8)


# ──────────────────────────────────────────────────────────────────────
# 15. BalaGoyalWeighted — chained to_networkx + remove_self_loop
# ──────────────────────────────────────────────────────────────────────

class TestBalaGoyalWeightedPattern:
    """Tests the BalaGoyalWeightedOp pattern of chaining DGL ops."""

    def test_to_networkx_remove_self_loop_chain(self):
        """dgl.to_networkx(dgl.remove_self_loop(graph)) must work."""
        g = ptgraph.graph(make_triangle_with_self_loop())
        g.ndata['h'] = torch.randn(3, 4)

        g_clean = ptgraph.remove_self_loop(g)
        G = ptgraph.to_networkx(g_clean)

        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 3  # self-loop removed

    def test_degree_centrality_pipeline(self):
        """Full BalaGoyalWeightedOp init pattern."""
        g = ptgraph.graph(make_triangle_with_self_loop())

        G = ptgraph.to_networkx(ptgraph.remove_self_loop(g))
        centrality = nx.degree_centrality(G)
        weights = torch.tensor(list(centrality.values()))

        assert weights.shape == (3,)
        # All nodes in a triangle have same centrality
        assert allclose(weights, weights[0].expand(3))

    def test_centrality_weighted_beliefs(self):
        """Beliefs weighted by centrality — full pattern."""
        # Star graph: node 0 has highest centrality
        g = ptgraph.graph(([0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]))
        size = (g.num_nodes(),)

        G = ptgraph.to_networkx(ptgraph.remove_self_loop(g))
        centrality = nx.degree_centrality(G)
        weights = torch.tensor(list(centrality.values()))

        g.ndata['beliefs'] = torch.ones(size) * weights
        # Node 0 should have highest belief (highest centrality)
        assert g.ndata['beliefs'][0] > g.ndata['beliefs'][1]


# ──────────────────────────────────────────────────────────────────────
# 16. Full BalaGoyal pipeline integration test
# ──────────────────────────────────────────────────────────────────────

class TestBalaGoyalIntegration:
    """End-to-end test mimicking the full BalaGoyalOp flow."""

    def test_full_flow(self):
        torch.manual_seed(42)

        # Build a cycle graph from networkx (undirected → doubled edges)
        G_nx = nx.cycle_graph(5)
        g = ptgraph.from_networkx(G_nx)
        assert g.num_edges() == 10  # 5 undirected edges × 2

        n = g.num_nodes()
        epsilon = 0.1
        trials = 10

        # Init
        g.ndata['beliefs'] = torch.full((n,), 0.5)
        probs = torch.full((n,), 0.5) + epsilon
        count = torch.zeros(n) + trials
        sampler = torch.distributions.binomial.Binomial(
            total_count=count, probs=probs
        )
        g.ndata['logits'] = sampler.logits

        for step in range(3):
            # Experiment
            mask = g.ndata['beliefs'] > 0.5
            result = torch.stack((sampler.sample(), sampler.total_count))
            mask_2d = mask.tile((2, 1))
            result = result * mask_2d
            g.ndata['payoffs'] = result.T

            # Filter
            def filterfn(edges):
                return torch.gt(edges.src['payoffs'][:, 1], 0.0)

            edges = g.filter_edges(filterfn)
            if edges.shape[0] == 0:
                continue

            # Message, reduce, apply
            def messagefn(edges):
                return {'payoffs': edges.src['payoffs']}

            def reducefn(nodes):
                return {'payoffs': torch.sum(nodes.mailbox['payoffs'], dim=1)}

            def applyfn(nodes):
                logits = nodes.data['logits']
                values = nodes.data['payoffs'][:, 0]
                t = nodes.data['payoffs'][:, 1]
                prior = nodes.data['beliefs']
                # Simplified Bayes (not exact, just testing the pipeline)
                likelihood = values / t.clamp(min=1)
                posterior = prior * likelihood / (prior * likelihood + (1 - prior) * (1 - likelihood)).clamp(min=1e-8)
                return {'beliefs': posterior}

            g.send_and_recv(edges, messagefn, reducefn, applyfn)

        # Beliefs should still be valid probabilities
        assert (g.ndata['beliefs'] >= 0).all()
        assert (g.ndata['beliefs'] <= 1).all()


# ──────────────────────────────────────────────────────────────────────
# 17. Unreliable network ops patterns
# ──────────────────────────────────────────────────────────────────────

class TestUnreliableOpsPatterns:
    """
    Tests patterns from the Unreliable/Aligned/Unaligned ops:
    - Mixed-dimension mailbox (1D reliability + 2D payoffs)
    - Multi-condition filter combining edges.src fields
    - Scalar captured in reduce closure
    - Per-neighbor iteration with 1D mailbox key
    """

    def _make_unreliable_graph(self):
        """5-node cycle with reliability and payoffs."""
        g = ptgraph.graph(([0, 1, 2, 3, 4], [1, 2, 3, 4, 0]))
        n = g.num_nodes()
        g.ndata['beliefs'] = torch.full((n,), 0.6)
        g.ndata['logits'] = torch.zeros(n)
        g.ndata['reliability'] = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0])
        g.ndata['payoffs'] = torch.tensor([
            [3.0, 5.0],
            [2.0, 5.0],
            [4.0, 5.0],  # unreliable node
            [0.0, 0.0],  # no evidence
            [1.0, 5.0],  # unreliable node
        ])
        return g

    def test_mixed_dim_mailbox(self):
        """
        AlignedOp sends 2D payoffs + 1D reliability.
        Mailbox must be payoffs: [N, deg, 2], reliability: [N, deg].
        """
        # 0→2, 1→2 — node 2 has 2 incoming
        g = ptgraph.graph(([0, 1], [2, 2]))
        g.ndata['payoffs'] = torch.tensor([[3.0, 5.0], [2.0, 5.0], [0.0, 0.0]])
        g.ndata['reliability'] = torch.tensor([1.0, 0.5, 0.0])
        g.ndata['beliefs'] = torch.tensor([0.6, 0.7, 0.5])

        shapes = {}

        def msg_fn(edges):
            return {
                'payoffs': edges.src['payoffs'],
                'reliability': edges.src['reliability'],
            }

        def reduce_fn(nodes):
            shapes['payoffs'] = nodes.mailbox['payoffs'].shape
            shapes['reliability'] = nodes.mailbox['reliability'].shape
            # Iterate per neighbor like AlignedOp
            _, neighbours = nodes.mailbox['reliability'].shape
            prior = nodes.data['beliefs']
            for i in range(neighbours):
                values = nodes.mailbox['payoffs'][:, i, 0]
                trials = nodes.mailbox['payoffs'][:, i, 1]
                rel = nodes.mailbox['reliability'][:, i]
                # Simplified update
                prior = prior + values / trials.clamp(min=1) * rel * 0.01
            return {'beliefs': prior}

        g.send_and_recv(torch.arange(2), msg_fn, reduce_fn)

        # Check shapes: node 2 has deg=2
        assert shapes['payoffs'] == (1, 2, 2)   # [num_nodes, deg, 2]
        assert shapes['reliability'] == (1, 2)    # [num_nodes, deg]
        # Node 2 updated, others preserved
        assert g.ndata['beliefs'][2].item() != 0.5
        assert g.ndata['beliefs'][0].item() == pytest.approx(0.6)

    def test_mixed_dim_varying_degrees(self):
        """
        Mixed-dimension mailbox with different in-degrees across nodes.
        Ensures degree bucketing handles 1D and 2D keys in same call.
        """
        # 0→2 (deg 1), 0→3, 1→3 (deg 2)
        g = ptgraph.graph(([0, 0, 1], [2, 3, 3]))
        g.ndata['payoffs'] = torch.tensor([
            [3.0, 5.0], [2.0, 5.0], [0.0, 0.0], [0.0, 0.0],
        ])
        g.ndata['reliability'] = torch.tensor([1.0, 0.5, 0.0, 0.0])
        g.ndata['beliefs'] = torch.tensor([0.6, 0.7, 0.5, 0.5])

        bucket_shapes = []

        def msg_fn(edges):
            return {
                'payoffs': edges.src['payoffs'],
                'reliability': edges.src['reliability'],
            }

        def reduce_fn(nodes):
            bucket_shapes.append({
                'payoffs': nodes.mailbox['payoffs'].shape,
                'reliability': nodes.mailbox['reliability'].shape,
            })
            return {'beliefs': nodes.data['beliefs'] + 0.1}

        g.send_and_recv(torch.arange(3), msg_fn, reduce_fn)

        # Should have 2 buckets: deg=1 (node 2) and deg=2 (node 3)
        assert len(bucket_shapes) == 2
        all_payoff_shapes = {s['payoffs'] for s in bucket_shapes}
        all_rel_shapes = {s['reliability'] for s in bucket_shapes}
        assert (1, 1, 2) in all_payoff_shapes  # deg=1
        assert (1, 2, 2) in all_payoff_shapes  # deg=2
        assert (1, 1) in all_rel_shapes        # deg=1
        assert (1, 2) in all_rel_shapes        # deg=2

    def test_ideal_op_filter_multi_condition(self):
        """
        UnreliableNetworkIdealOp filter: (payoffs > 0) * reliability → bool.
        Combines two edges.src fields with element-wise multiplication.
        """
        g = self._make_unreliable_graph()

        def filterfn(edges):
            reliability = edges.src['reliability']
            payoffs = edges.src['payoffs']
            return (torch.gt(payoffs[:, 1], 0.0) * reliability).bool()

        kept = g.filter_edges(filterfn)

        # Edges: 0→1, 1→2, 2→3, 3→4, 4→0
        # Node 0: payoffs>0 AND reliable=1 → edge 0→1 KEPT
        # Node 1: payoffs>0 AND reliable=1 → edge 1→2 KEPT
        # Node 2: payoffs>0 BUT reliable=0 → edge 2→3 FILTERED
        # Node 3: payoffs=0               → edge 3→4 FILTERED
        # Node 4: payoffs>0 BUT reliable=0 → edge 4→0 FILTERED
        assert allclose(kept, torch.tensor([0, 1]))

    def test_aligned_op_full_pattern(self):
        """
        Full AlignedOp reduce: per-neighbor iteration with reliability
        from mailbox, updating prior sequentially.
        """
        g = ptgraph.graph(([0, 1, 2], [3, 3, 3]))
        n = g.num_nodes()
        g.ndata['payoffs'] = torch.tensor([
            [3.0, 5.0], [4.0, 5.0], [2.0, 5.0], [0.0, 0.0],
        ])
        g.ndata['reliability'] = torch.tensor([1.0, 0.8, 0.0, 0.0])
        g.ndata['beliefs'] = torch.tensor([0.6, 0.7, 0.5, 0.5])
        g.ndata['logits'] = torch.zeros(n)

        def msg_fn(edges):
            return {
                'payoffs': edges.src['payoffs'],
                'reliability': edges.src['reliability'],
            }

        def reduce_fn(nodes):
            logits = nodes.data['logits']
            prior = nodes.data['beliefs']
            _, neighbours = nodes.mailbox['reliability'].shape
            for i in range(neighbours):
                values = nodes.mailbox['payoffs'][:, i, 0]
                trials = nodes.mailbox['payoffs'][:, i, 1]
                reliability = nodes.mailbox['reliability'][:, i]
                # Simplified jeffrey-like update
                likelihood = values / trials.clamp(min=1)
                update = (likelihood - 0.5) * reliability * 0.1
                prior = prior + update
            return {'beliefs': prior}

        g.send_and_recv(torch.arange(3), msg_fn, reduce_fn)

        # Node 3 received from all 3, should be updated
        assert g.ndata['beliefs'][3].item() != 0.5
        # Unreliable source (node 2, rel=0) should contribute nothing
        # So result should be driven by nodes 0 and 1 only

    def test_unaligned_op_trust_key(self):
        """
        UnalignedOp sends 'trust' instead of 'reliability'.
        Same pattern, different key name.
        """
        g = ptgraph.graph(([0, 1], [2, 2]))
        g.ndata['payoffs'] = torch.tensor([[3.0, 5.0], [2.0, 5.0], [0.0, 0.0]])
        g.ndata['trust'] = torch.tensor([0.9, 0.3, 0.0])
        g.ndata['beliefs'] = torch.tensor([0.6, 0.7, 0.5])
        g.ndata['logits'] = torch.zeros(3)

        def msg_fn(edges):
            return {'payoffs': edges.src['payoffs'], 'trust': edges.src['trust']}

        def reduce_fn(nodes):
            _, neighbours = nodes.mailbox['trust'].shape
            prior = nodes.data['beliefs']
            for i in range(neighbours):
                trust = nodes.mailbox['trust'][:, i]
                values = nodes.mailbox['payoffs'][:, i, 0]
                trials = nodes.mailbox['payoffs'][:, i, 1]
                likelihood = values / trials.clamp(min=1)
                prior = prior + (likelihood - 0.5) * trust * 0.1
            return {'beliefs': prior}

        g.send_and_recv(torch.arange(2), msg_fn, reduce_fn)
        assert g.ndata['beliefs'][2].item() != 0.5
        assert g.ndata['beliefs'][0].item() == pytest.approx(0.6)

    def test_modified_aligned_scalar_in_closure(self):
        """
        ModifiedAlignedOp captures self._network_reliability (a scalar)
        inside the reduce closure. No per-neighbor loop.
        """
        g = ptgraph.graph(([0, 1, 2], [3, 3, 3]))
        n = g.num_nodes()
        g.ndata['payoffs'] = torch.tensor([
            [3.0, 5.0], [4.0, 5.0], [2.0, 5.0], [0.0, 0.0],
        ])
        g.ndata['beliefs'] = torch.tensor([0.6, 0.7, 0.5, 0.5])
        g.ndata['logits'] = torch.zeros(n)

        # Simulating the closure capturing a scalar
        network_reliability = 0.8

        def msg_fn(edges):
            return {
                'payoffs': edges.src['payoffs'],
                'reliability': edges.src.get('reliability', torch.ones(len(edges))),
            }

        def reduce_fn(nodes):
            prior = nodes.data['beliefs']
            # Aggregate across ALL neighbors at once (no for loop)
            agg_values = torch.sum(nodes.mailbox['payoffs'][:, :, 0], dim=1)
            agg_trials = torch.sum(nodes.mailbox['payoffs'][:, :, 1], dim=1)
            likelihood = agg_values / agg_trials.clamp(min=1)
            # Use captured scalar
            update = (likelihood - 0.5) * network_reliability * 0.1
            posterior = prior + update
            return {'beliefs': posterior}

        g.send_and_recv(torch.arange(3), msg_fn, reduce_fn)
        assert g.ndata['beliefs'][3].item() != 0.5

    def test_aggregated_mailbox_slicing(self):
        """
        ModifiedAlignedOp uses mailbox[:, :, 0] and [:, :, 1] to aggregate
        across all neighbors. Verify the slicing works correctly.
        """
        # 0→2, 1→2
        g = ptgraph.graph(([0, 1], [2, 2]))
        g.ndata['payoffs'] = torch.tensor([[3.0, 5.0], [4.0, 10.0], [0.0, 0.0]])
        g.ndata['beliefs'] = torch.zeros(3)

        def msg_fn(edges):
            return {'payoffs': edges.src['payoffs']}

        def reduce_fn(nodes):
            agg_values = torch.sum(nodes.mailbox['payoffs'][:, :, 0], dim=1)
            agg_trials = torch.sum(nodes.mailbox['payoffs'][:, :, 1], dim=1)
            return {'agg_values': agg_values, 'agg_trials': agg_trials}

        g.send_and_recv(torch.arange(2), msg_fn, reduce_fn)
        # Node 2: values = 3+4 = 7, trials = 5+10 = 15
        assert g.ndata['agg_values'][2].item() == pytest.approx(7.0)
        assert g.ndata['agg_trials'][2].item() == pytest.approx(15.0)

    def test_unreliable_filter_then_send_recv(self):
        """Full pipeline: filter by reliability, then send_and_recv."""
        g = self._make_unreliable_graph()

        # Filter: only reliable nodes with evidence
        def filterfn(edges):
            return (torch.gt(edges.src['payoffs'][:, 1], 0.0) *
                    edges.src['reliability']).bool()

        edges = g.filter_edges(filterfn)

        def msg_fn(edges):
            return {'payoffs': edges.src['payoffs']}

        def reduce_fn(nodes):
            return {'payoffs': torch.sum(nodes.mailbox['payoffs'], dim=1)}

        def apply_fn(nodes):
            return {'beliefs': nodes.data['beliefs'] * 0.9 + 0.1}

        if edges.shape[0] > 0:
            g.send_and_recv(edges, msg_fn, reduce_fn, apply_fn)

        # Only nodes receiving from reliable sources get updated
        # Edge 0→1 kept (node 0 reliable, has payoffs)
        # Edge 1→2 kept (node 1 reliable, has payoffs)
        # Others filtered
        # So node 1 and node 2 receive messages
        assert g.ndata['beliefs'][1].item() == pytest.approx(0.64)  # 0.6*0.9+0.1
        assert g.ndata['beliefs'][2].item() == pytest.approx(0.64)
        # Node 0 didn't receive (edge 4→0 filtered) → unchanged
        assert g.ndata['beliefs'][0].item() == pytest.approx(0.6)


# ──────────────────────────────────────────────────────────────────────
# 18. Comparison against real DGL (optional)
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_DGL, reason="Real DGL not installed")
class TestVsRealDGL:
    """Side-by-side comparison with real DGL."""

    def _compare_send_recv(self, src, dst, h, msg_fn_ptgraph, msg_fn_dgl,
                           reduce_ptgraph, reduce_dgl, edge_subset=None):
        """Run send_and_recv on both and compare."""
        # ptgraph
        g1 = ptgraph.graph((src, dst))
        g1.ndata['h'] = h.clone()
        eid1 = edge_subset if edge_subset is not None else torch.arange(g1.num_edges())
        g1.send_and_recv(eid1, msg_fn_ptgraph, reduce_ptgraph)

        # Real DGL
        g2 = _real_dgl.graph((src, dst))
        g2.ndata['h'] = h.clone()
        eid2 = edge_subset if edge_subset is not None else g2.edges(form='eid')
        g2.send_and_recv(eid2, msg_fn_dgl, reduce_dgl)

        # Compare
        for key in g2.ndata:
            if key in g1.ndata:
                assert allclose(g1.ndata[key], g2.ndata[key]), \
                    f"Mismatch in ndata['{key}']:\nptgraph: {g1.ndata[key]}\nDGL:     {g2.ndata[key]}"

    def test_sum_reduce(self):
        import dgl.function as fn
        src, dst = [0, 1, 2, 0], [1, 2, 0, 2]
        h = torch.randn(3, 4)
        self._compare_send_recv(
            src, dst, h,
            lambda e: {'m': e.src['h']},
            fn.copy_u('h', 'm'),
            'sum',
            fn.sum('m', 'm'),
        )

    def test_udf_mean(self):
        src, dst = [0, 1, 2, 0], [1, 2, 0, 2]
        h = torch.randn(3, 4)
        reduce_fn = lambda n: {'m': torch.mean(n.mailbox['m'], dim=1)}
        self._compare_send_recv(
            src, dst, h,
            lambda e: {'m': e.src['h']},
            lambda e: {'m': e.src['h']},
            reduce_fn, reduce_fn,
        )

    def test_from_networkx_undirected(self):
        G = nx.karate_club_graph()
        g1 = ptgraph.from_networkx(G)
        g2 = _real_dgl.from_networkx(G)
        assert g1.num_nodes() == g2.num_nodes()
        assert g1.num_edges() == g2.num_edges()

    def test_non_recv_preserved(self):
        src, dst = [0], [1]
        h = torch.randn(3, 4)

        g1 = ptgraph.graph((src, dst), num_nodes=3)
        g1.ndata['h'] = h.clone()
        g1.ndata['m'] = torch.full((3, 4), 99.0)
        g1.send_and_recv(
            torch.tensor([0]),
            lambda e: {'m': e.src['h']},
            'sum',
        )

        g2 = _real_dgl.graph((src, dst), num_nodes=3)
        g2.ndata['h'] = h.clone()
        g2.ndata['m'] = torch.full((3, 4), 99.0)
        import dgl.function as fn
        g2.send_and_recv(
            torch.tensor([0]),
            fn.copy_u('h', 'm'),
            fn.sum('m', 'm'),
        )

        for i in range(3):
            assert allclose(g1.ndata['m'][i], g2.ndata['m'][i]), \
                f"Node {i} mismatch:\nptgraph: {g1.ndata['m'][i]}\nDGL:     {g2.ndata['m'][i]}"


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--standalone" in args:
        args.remove("--standalone")
        args.extend(["-k", "not TestVsRealDGL"])
        print("Running in standalone mode (skipping DGL comparison tests)")
    elif HAS_DGL:
        print(f"Real DGL detected — running comparison tests too")
    else:
        print("Real DGL not found — comparison tests will be skipped")
    print()

    sys.exit(pytest.main([__file__, "-v", "--tb=short"] + args))
