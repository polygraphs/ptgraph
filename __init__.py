"""
Pure PyTorch replacement for a subset of DGL functionality.

Replaces:
  - dgl.graph(edges)
  - dgl.to_networkx(graph)
  - dgl.remove_self_loop(graph)
  - graph.filter_edges(predicate)
  - graph.send_and_recv(edges, message_fn, reduce_fn)

Dependencies: torch, networkx
Optional: dgl (only needed for one-time .bin conversion)
"""

from __future__ import annotations

import torch
import networkx as nx
from torch import Tensor
from typing import Callable, Dict, Optional, Tuple, Union, List, Any
from pathlib import Path

# ---------------------------------------------------------------------------
# Edge / Node view objects (mimic DGL's EdgeBatch / NodeBatch)
# ---------------------------------------------------------------------------


class EdgeView:
    """Passed to user-defined message functions. Mimics DGL's ``edges``."""

    def __init__(
        self,
        src_ids: Tensor,
        dst_ids: Tensor,
        edge_ids: Tensor,
        src_data: Dict[str, Tensor],
        dst_data: Dict[str, Tensor],
        edge_data: Dict[str, Tensor],
    ):
        self._src_ids = src_ids
        self._dst_ids = dst_ids
        self._edge_ids = edge_ids
        self.src = src_data  # {feat_name: tensor[num_msg_edges, *]}
        self.dst = dst_data
        self.data = edge_data

    def src_ids(self) -> Tensor:
        return self._src_ids

    def dst_ids(self) -> Tensor:
        return self._dst_ids

    def edge_ids(self) -> Tensor:
        return self._edge_ids

    def __len__(self) -> int:
        return self._src_ids.shape[0]


class NodeView:
    """Passed to user-defined apply functions. Mimics DGL's ``nodes``."""

    def __init__(self, node_data: Dict[str, Tensor]):
        self.data = node_data

    def __len__(self) -> int:
        for v in self.data.values():
            return v.shape[0]
        return 0


class NodeMailbox:
    """Passed to user-defined reduce functions. Mimics DGL's ``nodes``."""

    def __init__(self, mailbox: Dict[str, Tensor], node_data: Dict[str, Tensor]):
        self.mailbox = mailbox  # {msg_name: tensor[num_recv_nodes, deg, *]}
        self.data = node_data

    def __len__(self) -> int:
        for v in self.data.values():
            return v.shape[0]
        for v in self.mailbox.values():
            return v.shape[0]
        return 0


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


class Graph:
    """
    A simple homogeneous directed graph stored in COO (edge-list) format.

    Parameters
    ----------
    edges : tuple of (Tensor|list, Tensor|list)
        ``(src, dst)`` – 1-D integer tensors/lists of the same length.
    num_nodes : int, optional
        If omitted, inferred as ``max(src.max(), dst.max()) + 1``.
    device : torch.device, optional
    """

    def __init__(
        self,
        edges: Tuple[Union[Tensor, list], Union[Tensor, list]],
        num_nodes: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        src, dst = edges
        if not isinstance(src, Tensor):
            src = torch.tensor(src, dtype=torch.long)
        if not isinstance(dst, Tensor):
            dst = torch.tensor(dst, dtype=torch.long)
        src = src.to(dtype=torch.long)
        dst = dst.to(dtype=torch.long)
        assert src.ndim == 1 and dst.ndim == 1 and src.shape == dst.shape

        if device is not None:
            src, dst = src.to(device), dst.to(device)

        self._src = src
        self._dst = dst
        self._num_nodes = (
            num_nodes
            if num_nodes is not None
            else (int(max(src.max(), dst.max())) + 1 if src.numel() > 0 else 0)
        )
        self.ndata: Dict[str, Tensor] = {}  # node features
        self.edata: Dict[str, Tensor] = {}  # edge features

    # -- properties ----------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self._src.device

    def num_nodes(self) -> int:
        return self._num_nodes

    def num_edges(self) -> int:
        return self._src.shape[0]

    def nodes(self) -> Tensor:
        """Return all node IDs."""
        return torch.arange(self._num_nodes, device=self.device)

    def number_of_nodes(self) -> int:
        """Alias for ``num_nodes()``."""
        return self._num_nodes

    def number_of_edges(self) -> int:
        """Alias for ``num_edges()``."""
        return self._src.shape[0]

    def in_degrees(self, v: Optional[Union[Tensor, int]] = None) -> Tensor:
        """Return in-degrees. If *v* is given, return only for those nodes."""
        deg = torch.zeros(self._num_nodes, dtype=torch.long, device=self.device)
        ones = torch.ones(self.num_edges(), dtype=torch.long, device=self.device)
        deg.scatter_add_(0, self._dst, ones)
        if v is not None:
            return deg[v]
        return deg

    def out_degrees(self, u: Optional[Union[Tensor, int]] = None) -> Tensor:
        """Return out-degrees. If *u* is given, return only for those nodes."""
        deg = torch.zeros(self._num_nodes, dtype=torch.long, device=self.device)
        ones = torch.ones(self.num_edges(), dtype=torch.long, device=self.device)
        deg.scatter_add_(0, self._src, ones)
        if u is not None:
            return deg[u]
        return deg

    def has_edges_between(self, u: Tensor, v: Tensor) -> Tensor:
        """Check whether edges exist between pairs of nodes."""
        u = torch.as_tensor(u, dtype=torch.long, device=self.device)
        v = torch.as_tensor(v, dtype=torch.long, device=self.device)
        if self._edge_keys_sorted is None:
            keys = self._src * self._num_nodes + self._dst
            self._edge_keys_sorted, _ = torch.sort(keys)
        query = u * self._num_nodes + v
        idx = torch.searchsorted(self._edge_keys_sorted, query)
        idx = idx.clamp(max=self._edge_keys_sorted.numel() - 1)
        return self._edge_keys_sorted[idx] == query

    def edges(
        self, form: str = "uv"
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
        """Return edges. ``form`` is 'uv' or 'all'."""
        if form == "uv":
            return self._src, self._dst
        elif form == "all":
            return (
                self._src,
                self._dst,
                torch.arange(self.num_edges(), device=self.device),
            )
        raise ValueError(f"Unknown form: {form}")

    def to(self, device: torch.device) -> "Graph":
        """Move graph (and all features) to *device*."""
        g = Graph((self._src, self._dst), num_nodes=self._num_nodes, device=device)
        g.ndata = {k: v.to(device) for k, v in self.ndata.items()}
        g.edata = {k: v.to(device) for k, v in self.edata.items()}
        return g

    # -- filter_edges --------------------------------------------------------

    def filter_edges(self, predicate: Callable[[EdgeView], Tensor]) -> Tensor:
        """
        Return edge IDs for which *predicate* returns True.

        Parameters
        ----------
        predicate : callable
            ``(EdgeView) -> BoolTensor[num_edges]``

        Returns
        -------
        Tensor of edge indices (int64).
        """
        edge_ids = torch.arange(self.num_edges(), device=self.device)
        src_data = {k: v[self._src] for k, v in self.ndata.items()}
        dst_data = {k: v[self._dst] for k, v in self.ndata.items()}
        ev = EdgeView(
            src_ids=self._src,
            dst_ids=self._dst,
            edge_ids=edge_ids,
            src_data=src_data,
            dst_data=dst_data,
            edge_data=self.edata,
        )
        mask = predicate(ev)
        return edge_ids[mask]

    # -- send_and_recv -------------------------------------------------------

    def send_and_recv(
        self,
        edges: Union[Tensor, Tuple[Tensor, Tensor]],
        message_fn: Callable[[EdgeView], Dict[str, Tensor]],
        reduce_fn: Union[str, Callable[[NodeMailbox], Dict[str, Tensor]]],
        apply_node_fn: Optional[
            Callable[[Dict[str, Tensor]], Dict[str, Tensor]]
        ] = None,
    ) -> None:
        """
        Message-passing on a subset of edges.

        Parameters
        ----------
        edges : Tensor | tuple(Tensor, Tensor)
            Either a 1-D tensor of edge indices **or** ``(src, dst)`` tensors
            identifying the edges to activate.
        message_fn : callable
            ``(EdgeView) -> dict[str, Tensor]``  — produces per-edge messages.
        reduce_fn : str or callable
            Built-in strings: ``'sum'``, ``'mean'``, ``'max'``, ``'min'``.
            Or a callable ``(NodeMailbox) -> dict[str, Tensor]``.
        apply_node_fn : callable, optional
            ``(node_data_dict) -> node_data_dict`` applied after reduce.
        """
        # --- resolve edge indices -------------------------------------------
        if isinstance(edges, tuple):
            src_sel, dst_sel = edges
            if not isinstance(src_sel, Tensor):
                src_sel = torch.tensor(src_sel, dtype=torch.long, device=self.device)
            if not isinstance(dst_sel, Tensor):
                dst_sel = torch.tensor(dst_sel, dtype=torch.long, device=self.device)
            edge_ids = torch.arange(src_sel.shape[0], device=self.device)
            edge_data = {}
        else:
            edge_ids = edges.to(self.device)
            src_sel = self._src[edge_ids]
            dst_sel = self._dst[edge_ids]
            edge_data = {k: v[edge_ids] for k, v in self.edata.items()}

        # --- message phase ---------------------------------------------------
        src_data = {k: v[src_sel] for k, v in self.ndata.items()}
        dst_data = {k: v[dst_sel] for k, v in self.ndata.items()}
        ev = EdgeView(src_sel, dst_sel, edge_ids, src_data, dst_data, edge_data)
        messages: Dict[str, Tensor] = message_fn(ev)

        # --- handle empty messages (e.g. NoOp returning {}) -----------------
        if not messages:
            # Nothing to reduce; still run apply if given
            if apply_node_fn is not None:
                unique_dst = torch.unique(dst_sel)
                recv_ndata = {k: v[unique_dst] for k, v in self.ndata.items()}
                nv = NodeView(recv_ndata)
                updated = apply_node_fn(nv)
                if isinstance(updated, dict):
                    for k, v in updated.items():
                        if k not in self.ndata:
                            self.ndata[k] = torch.zeros(
                                self._num_nodes,
                                *v.shape[1:],
                                dtype=v.dtype,
                                device=self.device,
                            )
                        self.ndata[k][unique_dst] = v
            return

        # --- track receiving nodes -------------------------------------------
        unique_dst = torch.unique(dst_sel)

        # --- reduce phase ----------------------------------------------------
        if isinstance(reduce_fn, str):
            self._builtin_reduce(dst_sel, messages, reduce_fn)
        else:
            self._udf_reduce(dst_sel, messages, reduce_fn, edge_ids)

        # --- apply phase (DGL semantics: merge ndata + reduce results,
        #     run only on receiving nodes, write back) ------------------------
        if apply_node_fn is not None:
            # DGL merges original node features with reduce results
            recv_ndata = {}
            for k, v in self.ndata.items():
                recv_ndata[k] = v[unique_dst]
            nv = NodeView(recv_ndata)
            updated = apply_node_fn(nv)
            if isinstance(updated, dict):
                for k, v in updated.items():
                    if k not in self.ndata:
                        self.ndata[k] = torch.zeros(
                            self._num_nodes,
                            *v.shape[1:],
                            dtype=v.dtype,
                            device=self.device,
                        )
                    self.ndata[k][unique_dst] = v

    # helpers for reduce

    def _builtin_reduce(
        self, dst: Tensor, messages: Dict[str, Tensor], op: str
    ) -> None:
        """Scatter-based built-in reduce: sum | mean | max | min.

        Only destination nodes that receive messages are updated;
        all other nodes retain their previous values (DGL semantics).
        """
        recv_mask = torch.zeros(self._num_nodes, dtype=torch.bool, device=self.device)
        recv_mask.scatter_(0, dst, True)

        for key, msg in messages.items():
            # Start from existing values if present, else zeros
            if key in self.ndata:
                out = self.ndata[key].clone()
            else:
                out = torch.zeros(
                    self._num_nodes, *msg.shape[1:], dtype=msg.dtype, device=self.device
                )

            idx = dst.unsqueeze(-1).expand_as(msg) if msg.ndim > 1 else dst

            if op == "sum":
                # Zero out receiving nodes first, then accumulate
                out[recv_mask] = 0
                out.scatter_add_(0, idx, msg)
            elif op == "mean":
                out[recv_mask] = 0
                out.scatter_add_(0, idx, msg)
                count = torch.zeros(
                    self._num_nodes, device=self.device, dtype=msg.dtype
                )
                ones = torch.ones(dst.shape[0], device=self.device, dtype=msg.dtype)
                count.scatter_add_(0, dst, ones)
                count = count.clamp(min=1)
                # Only divide receiving nodes
                if out.ndim > 1:
                    out[recv_mask] = out[recv_mask] / count[recv_mask].unsqueeze(-1)
                else:
                    out[recv_mask] = out[recv_mask] / count[recv_mask]
            elif op in ("max", "min"):
                fill = float("-inf") if op == "max" else float("inf")
                reduced = torch.full_like(out, fill)
                reduced.scatter_reduce_(
                    0, idx, msg, reduce=f"a{op}", include_self=False
                )
                out[recv_mask] = reduced[recv_mask]
            else:
                raise ValueError(f"Unknown reduce op: {op}")
            self.ndata[key] = out

    def _udf_reduce(
        self,
        dst: Tensor,
        messages: Dict[str, Tensor],
        reduce_fn: Callable,
        edge_ids: Tensor,
    ) -> None:
        """
        User-defined reduce using degree bucketing (matching DGL semantics).

        DGL groups destination nodes by their in-degree, then calls the
        reduce UDF once per group with a mailbox shaped exactly
        ``[num_nodes_in_bucket, degree, *feat_shape]`` — **no zero-padding**.
        Messages within each node's mailbox are sorted by edge ID.
        """
        unique_dst, inverse = torch.unique(dst, sorted=True, return_inverse=True)
        counts = torch.bincount(inverse, minlength=unique_dst.shape[0])

        # Sort messages by (destination_node, edge_id) so mailbox order
        # matches DGL's semantics of sorting by edge ID per node.
        max_eid = edge_ids.max().item() + 1 if edge_ids.numel() > 0 else 1
        sort_keys = inverse.long() * max_eid + edge_ids.long()
        sort_order = torch.argsort(sort_keys, stable=True)
        sorted_inverse = inverse[sort_order]
        sorted_msgs = {k: v[sort_order] for k, v in messages.items()}

        # --- degree bucketing (matching DGL's invoke_udf_reduce) -----------
        unique_degs = torch.unique(counts[counts > 0], sorted=True)

        for key, val in self.ndata.items():
            # ensure all ndata keys survive even if reduce doesn't touch them
            pass

        all_results: Dict[str, List] = {}
        all_nodes: List[Tensor] = []

        for deg in unique_degs:
            deg_int = int(deg.item())
            # Nodes in this bucket (indices into unique_dst)
            bucket_mask = counts == deg
            bucket_local = torch.where(bucket_mask)[0]  # indices into unique_dst
            bucket_global = unique_dst[bucket_local]  # original node IDs
            num_nodes_bkt = bucket_local.shape[0]

            # Gather messages for this bucket
            # Find which sorted messages belong to nodes in this bucket
            msg_mask = bucket_mask[sorted_inverse]
            msg_indices = torch.where(msg_mask)[0]

            mailbox: Dict[str, Tensor] = {}
            for key, msg in sorted_msgs.items():
                bkt_msg = msg[msg_indices]
                # Reshape to [num_nodes_bkt, deg, *feat_shape]
                mailbox[key] = bkt_msg.reshape(num_nodes_bkt, deg_int, *msg.shape[1:])

            node_data_bkt = {k: v[bucket_global] for k, v in self.ndata.items()}
            mb = NodeMailbox(mailbox, node_data_bkt)
            result = reduce_fn(mb)

            all_nodes.append(bucket_global)
            for key, val in result.items():
                if key not in all_results:
                    all_results[key] = []
                all_results[key].append((bucket_global, val))

        # Write back results
        for key, node_val_pairs in all_results.items():
            if key not in self.ndata:
                sample_val = node_val_pairs[0][1]
                self.ndata[key] = torch.zeros(
                    self._num_nodes,
                    *sample_val.shape[1:],
                    dtype=sample_val.dtype,
                    device=self.device,
                )
            for node_ids, val in node_val_pairs:
                self.ndata[key][node_ids] = val

    def update_all(
        self,
        message_fn: Callable[[EdgeView], Dict[str, Tensor]],
        reduce_fn: Union[str, Callable[[NodeMailbox], Dict[str, Tensor]]],
        apply_node_fn: Optional[Callable] = None,
    ) -> None:
        """
        Send messages along ALL edges and reduce at ALL nodes.

        Equivalent to ``send_and_recv(all_edges, ...)``.
        """
        self.send_and_recv(
            torch.arange(self.num_edges(), device=self.device),
            message_fn,
            reduce_fn,
            apply_node_fn,
        )

    def apply_nodes(
        self,
        func: Callable,
        v: Optional[Tensor] = None,
    ) -> None:
        """Apply function to nodes."""
        if v is None:
            v = torch.arange(self._num_nodes, device=self.device)
        elif not isinstance(v, Tensor):
            v = torch.tensor(v, dtype=torch.long, device=self.device)
        ndata_slice = {k: val[v] for k, val in self.ndata.items()}
        nv = NodeView(ndata_slice)
        result = func(nv)
        if isinstance(result, dict):
            for k, val in result.items():
                if k not in self.ndata:
                    self.ndata[k] = torch.zeros(
                        self._num_nodes,
                        *val.shape[1:],
                        dtype=val.dtype,
                        device=self.device,
                    )
                self.ndata[k][v] = val

    def apply_edges(
        self,
        func: Callable[[EdgeView], Dict[str, Tensor]],
        edges: Optional[Tensor] = None,
    ) -> None:
        """Apply function to edges."""
        if edges is None:
            eid = torch.arange(self.num_edges(), device=self.device)
        else:
            eid = edges.to(self.device)
        src_sel, dst_sel = self._src[eid], self._dst[eid]
        src_data = {k: v[src_sel] for k, v in self.ndata.items()}
        dst_data = {k: v[dst_sel] for k, v in self.ndata.items()}
        edge_data = {k: v[eid] for k, v in self.edata.items()}
        ev = EdgeView(src_sel, dst_sel, eid, src_data, dst_data, edge_data)
        result = func(ev)
        if isinstance(result, dict):
            for k, val in result.items():
                if k not in self.edata:
                    self.edata[k] = torch.zeros(
                        self.num_edges(),
                        *val.shape[1:],
                        dtype=val.dtype,
                        device=self.device,
                    )
                self.edata[k][eid] = val

    def __repr__(self) -> str:
        return (
            f"Graph(num_nodes={self._num_nodes}, num_edges={self.num_edges()}, "
            f"ndata_keys={list(self.ndata.keys())}, edata_keys={list(self.edata.keys())})"
        )


# ---------------------------------------------------------------------------
# Module-level functions (mirror dgl.*)
# ---------------------------------------------------------------------------


def graph(
    edges: Tuple[Union[Tensor, list], Union[Tensor, list]],
    num_nodes: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Graph:
    """Create a graph from an edge list.  Drop-in for ``dgl.graph(...)``."""
    return Graph(edges, num_nodes=num_nodes, device=device)


def to_networkx(
    g: Graph,
    node_attrs: Optional[List[str]] = None,
    edge_attrs: Optional[List[str]] = None,
) -> nx.DiGraph:
    """Convert to a NetworkX DiGraph.  Drop-in for ``dgl.to_networkx(...)``."""
    G = nx.DiGraph()
    G.add_nodes_from(range(g.num_nodes()))
    src, dst = g.edges()
    src_list, dst_list = src.cpu().tolist(), dst.cpu().tolist()

    for i, (u, v) in enumerate(zip(src_list, dst_list)):
        attr = {}
        if edge_attrs:
            for key in edge_attrs:
                attr[key] = g.edata[key][i].cpu()
        G.add_edge(u, v, **attr)

    if node_attrs:
        for key in node_attrs:
            vals = g.ndata[key].cpu()
            for nid in range(g.num_nodes()):
                G.nodes[nid][key] = vals[nid]

    return G


def from_networkx(
    nx_graph: nx.Graph,
    node_attrs: Optional[List[str]] = None,
    edge_attrs: Optional[List[str]] = None,
) -> Graph:
    """
    Create a Graph from a NetworkX graph.  Drop-in for ``dgl.from_networkx(...)``.

    Parameters
    ----------
    nx_graph : nx.Graph or nx.DiGraph
    node_attrs : list of str, optional
        Node attribute names to copy into ``g.ndata``.
    edge_attrs : list of str, optional
        Edge attribute names to copy into ``g.edata``.
    """
    # Relabel nodes to contiguous 0..N-1
    nx_graph = nx.convert_node_labels_to_integers(nx_graph, ordering="sorted")
    num_nodes = nx_graph.number_of_nodes()

    is_directed = nx_graph.is_directed()

    src_list, dst_list = [], []
    for u, v in nx_graph.edges():
        src_list.append(u)
        dst_list.append(v)
        if not is_directed:
            # DGL adds both directions for undirected graphs
            src_list.append(v)
            dst_list.append(u)

    if len(src_list) == 0:
        src_t = torch.zeros(0, dtype=torch.long)
        dst_t = torch.zeros(0, dtype=torch.long)
    else:
        src_t = torch.tensor(src_list, dtype=torch.long)
        dst_t = torch.tensor(dst_list, dtype=torch.long)

    g = Graph((src_t, dst_t), num_nodes=num_nodes)

    # Copy node attributes
    if node_attrs:
        for key in node_attrs:
            vals = [nx_graph.nodes[n].get(key) for n in range(num_nodes)]
            if isinstance(vals[0], Tensor):
                g.ndata[key] = torch.stack(vals)
            else:
                g.ndata[key] = torch.tensor(vals)

    # Copy edge attributes
    if edge_attrs:
        edges_for_attrs = list(nx_graph.edges())
        for key in edge_attrs:
            vals = []
            for u, v in edges_for_attrs:
                attr_val = nx_graph.edges[u, v].get(key)
                vals.append(attr_val)
                if not is_directed:
                    vals.append(attr_val)
            if isinstance(vals[0], Tensor):
                g.edata[key] = torch.stack(vals)
            else:
                g.edata[key] = torch.tensor(vals)

    return g


def remove_self_loop(g: Graph) -> Graph:
    """Return a new graph with self-loops removed.  Drop-in for ``dgl.remove_self_loop(...)``."""
    src, dst = g.edges()
    mask = src != dst
    new_g = Graph((src[mask], dst[mask]), num_nodes=g.num_nodes(), device=g.device)
    new_g.ndata = {k: v.clone() for k, v in g.ndata.items()}
    new_g.edata = {k: v[mask].clone() for k, v in g.edata.items()}
    return new_g


# ---------------------------------------------------------------------------
# Serialization — native format (.pt)
# ---------------------------------------------------------------------------

_FORMAT_VERSION = 1


def save_graphs(
    filepath: Union[str, Path],
    graphs: Union[Graph, List[Graph]],
    labels: Optional[Dict[str, Tensor]] = None,
) -> None:
    """
    Save one or more graphs (and optional labels) to a ``.pt`` file.

    Drop-in replacement for ``dgl.save_graphs(...)``.
    """
    if isinstance(graphs, Graph):
        graphs = [graphs]
    data = {
        "_format": "ptgraph",
        "_version": _FORMAT_VERSION,
        "graphs": [
            {
                "edges": (g._src.cpu(), g._dst.cpu()),
                "num_nodes": g.num_nodes(),
                "ndata": {k: v.cpu() for k, v in g.ndata.items()},
                "edata": {k: v.cpu() for k, v in g.edata.items()},
            }
            for g in graphs
        ],
        "labels": {k: v.cpu() for k, v in labels.items()} if labels else {},
    }
    torch.save(data, filepath)


def load_graphs(
    filepath: Union[str, Path],
    idx_list: Optional[List[int]] = None,
) -> Tuple[List[Graph], Dict[str, Tensor]]:
    """
    Load graphs (and labels) from a ``.pt`` file saved by :func:`save_graphs`.

    Drop-in replacement for ``dgl.load_graphs(...)``.

    Parameters
    ----------
    filepath : str or Path
    idx_list : list of int, optional
        Load only these graph indices.  Default: load all.

    Returns
    -------
    (list[Graph], dict[str, Tensor])
    """
    raw = torch.load(filepath, weights_only=False)

    # Detect format
    if raw.get("_format") == "ptgraph":
        entries = raw["graphs"]
        if idx_list is not None:
            entries = [entries[i] for i in idx_list]
        graphs = []
        for d in entries:
            g = Graph(d["edges"], num_nodes=d["num_nodes"])
            g.ndata.update(d["ndata"])
            g.edata.update(d["edata"])
            graphs.append(g)
        return graphs, raw.get("labels", {})

    raise ValueError(
        f"Unrecognised file format. Expected a file saved by save_graphs(). "
        f"If this is a DGL .bin file, use convert_from_dgl() first."
    )


# ---------------------------------------------------------------------------
# One-time DGL .bin → .pt conversion  (requires dgl installed)
# ---------------------------------------------------------------------------


def convert_from_dgl(
    src_path: Union[str, Path],
    dst_path: Optional[Union[str, Path]] = None,
    *,
    idx_list: Optional[List[int]] = None,
) -> Tuple[List[Graph], Dict[str, Tensor]]:
    """
    Convert a DGL ``.bin`` file to the native ``.pt`` format.

    Requires ``dgl`` to be installed.  Intended as a **one-time migration**
    step — after conversion you no longer need DGL.

    Parameters
    ----------
    src_path : path to the ``.bin`` file
    dst_path : path for the output ``.pt`` file.
        If ``None``, replaces the extension with ``.pt``.
    idx_list : optional list of graph indices to convert.

    Returns
    -------
    (list[Graph], dict[str, Tensor])  — the converted graphs + labels
    """
    try:
        import dgl as _dgl
    except ImportError:
        raise ImportError(
            "dgl is required for .bin conversion.  Install it in a "
            "temporary environment:\n  pip install dgl\n"
            "After converting your files you can uninstall it."
        )

    src_path = Path(src_path)
    if dst_path is None:
        dst_path = src_path.with_suffix(".pt")
    else:
        dst_path = Path(dst_path)

    dgl_graphs, dgl_labels = _dgl.load_graphs(str(src_path), idx_list)

    graphs: List[Graph] = []
    for dg in dgl_graphs:
        src, dst = dg.edges()
        g = Graph((src, dst), num_nodes=dg.num_nodes())
        for k, v in dg.ndata.items():
            g.ndata[k] = v
        for k, v in dg.edata.items():
            g.edata[k] = v
        graphs.append(g)

    labels = {}
    if dgl_labels:
        labels = {k: v for k, v in dgl_labels.items()}

    save_graphs(dst_path, graphs, labels)
    print(f"Converted {len(graphs)} graph(s): {src_path} → {dst_path}")
    return graphs, labels


# ---------------------------------------------------------------------------
# Batch conversion helper
# ---------------------------------------------------------------------------


def convert_all_from_dgl(
    directory: Union[str, Path],
    pattern: str = "*.bin",
    recursive: bool = False,
) -> List[Path]:
    """
    Batch-convert all DGL ``.bin`` files in a directory to ``.pt``.

    Returns list of output paths.
    """
    directory = Path(directory)
    glob_fn = directory.rglob if recursive else directory.glob
    converted = []
    for bin_path in sorted(glob_fn(pattern)):
        pt_path = bin_path.with_suffix(".pt")
        try:
            convert_from_dgl(bin_path, pt_path)
            converted.append(pt_path)
        except Exception as e:
            print(f"  FAILED {bin_path}: {e}")
    print(f"\nConverted {len(converted)} file(s).")
    return converted


# ---------------------------------------------------------------------------
# Misc utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Seed the torch RNG (used for all graph-level random operations)."""
    torch.manual_seed(seed)


def add_self_loop(g: Graph, edge_fill_value: float = 0.0) -> Graph:
    """Return a new graph with a self-loop added at every node.

    Edge features on new self-loop edges are filled with ``edge_fill_value``
    (default 0). If you use edge weights where 1 means "present", pass
    ``edge_fill_value=1.0``.
    """
    src, dst = g.edges()
    self_nodes = torch.arange(g.num_nodes(), dtype=torch.long, device=g.device)
    new_src = torch.cat([src, self_nodes])
    new_dst = torch.cat([dst, self_nodes])
    new_g = Graph((new_src, new_dst), num_nodes=g.num_nodes(), device=g.device)
    new_g.ndata = {k: v.clone() for k, v in g.ndata.items()}
    for k, v in g.edata.items():
        pad = torch.full(
            (g.num_nodes(), *v.shape[1:]),
            edge_fill_value,
            dtype=v.dtype,
            device=v.device,
        )
        new_g.edata[k] = torch.cat([v, pad])
    return new_g


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build a small triangle graph: 0→1, 1→2, 2→0, 0→0 (self-loop)
    g = graph(([0, 1, 2, 0], [1, 2, 0, 0]))
    print("Original:", g)

    # Assign features
    g.ndata["h"] = torch.randn(g.num_nodes(), 4)
    g.edata["w"] = torch.tensor([1.0, 2.0, 3.0, 0.5])

    # remove_self_loop
    g2 = remove_self_loop(g)
    print("No self-loops:", g2)
    assert g2.num_edges() == 3

    # filter_edges: keep edges where src < dst
    kept = g2.filter_edges(lambda e: e.src_ids() < e.dst_ids())
    print("Edges with src < dst:", kept)

    # send_and_recv with built-in sum reduce
    def msg_fn(edges: EdgeView) -> Dict[str, Tensor]:
        return {"m": edges.src["h"] * edges.data["w"].unsqueeze(-1)}

    g2.send_and_recv(
        torch.arange(g2.num_edges()),
        message_fn=msg_fn,
        reduce_fn="sum",
    )
    print("After send_and_recv, ndata keys:", list(g2.ndata.keys()))
    print("Aggregated 'm':\n", g2.ndata["m"])

    # to_networkx
    G = to_networkx(g2, node_attrs=["h"])
    print("NetworkX graph:", G.edges(), "nodes:", G.number_of_nodes())

    # save / load round-trip
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.pt")
        save_graphs(path, [g, g2], labels={"y": torch.tensor([0, 1])})
        loaded, labels = load_graphs(path)
        assert len(loaded) == 2
        assert loaded[0].num_edges() == 4
        assert loaded[1].num_edges() == 3
        assert torch.equal(labels["y"], torch.tensor([0, 1]))
        # selective loading
        loaded_one, _ = load_graphs(path, idx_list=[1])
        assert len(loaded_one) == 1
        assert loaded_one[0].num_edges() == 3
        print(f"Save/load round-trip OK ({path})")

    print("\n✓ All smoke tests passed.")
