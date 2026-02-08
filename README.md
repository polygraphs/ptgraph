Pure PyTorch replacement for a subset of DGL functionality.

Replaces:
  - dgl.graph(edges)
  - dgl.to_networkx(graph)
  - dgl.remove_self_loop(graph)
  - graph.filter_edges(predicate)
  - graph.send_and_recv(edges, message_fn, reduce_fn)

Dependencies: torch, networkx
