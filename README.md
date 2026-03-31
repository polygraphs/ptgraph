Pure PyTorch replacement for a subset of DGL functionality.

Replaces:
  - `dgl.graph(edges)`
  - `dgl.to_networkx(graph)`
  - `dgl.remove_self_loop(graph)`
  - `dgl.transforms.add_self_loop(graph)`
  - `graph.filter_edges(predicate)`
  - `graph.send_and_recv(edges, message_fn, reduce_fn)`

Dependencies: torch, networkx

## Comparison
DGL Version of Polygraphs running inside container:
```bash
$ container run \
    -v $(pwd)/configs:/configs \
    -v ~/polygraphs-cache:/app/polygraphs-cache \
    polygraphs run -f /configs/test.yaml
[MON] step 0001 Ksteps/s   0.00 A/B 0.56/0.44
[MON] step 0049 Ksteps/s   1.52 A/B 0.00/1.00
 INFO polygraphs> Sim #0001:     49 steps    0.04s; action: B undefined: 0 converged: 1 polarized: 0
[MON] step 0001 Ksteps/s   0.00 A/B 0.56/0.44
[MON] step 0073 Ksteps/s   1.66 A/B 0.00/1.00
 INFO polygraphs> Sim #0002:     73 steps    0.04s; action: B undefined: 0 converged: 1 polarized: 0
[MON] step 0001 Ksteps/s   0.00 A/B 0.56/0.44
[MON] step 0100 Ksteps/s   1.69 A/B 0.00/1.00
[MON] step 0107 Ksteps/s   1.65 A/B 0.00/1.00
 INFO polygraphs> Sim #0003:    107 steps    0.06s; action: B undefined: 0 converged: 1 polarized: 0
[MON] step 0001 Ksteps/s   0.00 A/B 0.44/0.56
[MON] step 0064 Ksteps/s   1.63 A/B 0.00/1.00
 INFO polygraphs> Sim #0004:     64 steps    0.04s; action: B undefined: 0 converged: 1 polarized: 0
[MON] step 0001 Ksteps/s   0.00 A/B 0.62/0.38
[MON] step 0093 Ksteps/s   1.71 A/B 0.00/1.00
 INFO polygraphs> Sim #0005:     93 steps    0.05s; action: B undefined: 0 converged: 1 polarized: 0
Bye.
```

ptgraph version of Polygraphs:

```bash
$ uv run run.py -f configs/test.yaml
[MON] step 0001 Ksteps/s   0.00 A/B 0.56/0.44
[MON] step 0049 Ksteps/s   2.58 A/B 0.00/1.00
 INFO polygraphs> Sim #0001:     49 steps    0.03s; action: B undefined: 0 converged: 1 polarized: 0
[MON] step 0001 Ksteps/s   0.00 A/B 0.56/0.44
[MON] step 0073 Ksteps/s   4.40 A/B 0.00/1.00
 INFO polygraphs> Sim #0002:     73 steps    0.02s; action: B undefined: 0 converged: 1 polarized: 0
[MON] step 0001 Ksteps/s   0.00 A/B 0.56/0.44
[MON] step 0100 Ksteps/s   4.48 A/B 0.00/1.00
[MON] step 0107 Ksteps/s   4.42 A/B 0.00/1.00
 INFO polygraphs> Sim #0003:    107 steps    0.02s; action: B undefined: 0 converged: 1 polarized: 0
[MON] step 0001 Ksteps/s   0.00 A/B 0.44/0.56
[MON] step 0064 Ksteps/s   4.31 A/B 0.00/1.00
 INFO polygraphs> Sim #0004:     64 steps    0.01s; action: B undefined: 0 converged: 1 polarized: 0
[MON] step 0001 Ksteps/s   0.00 A/B 0.62/0.38
[MON] step 0093 Ksteps/s   4.43 A/B 0.00/1.00
 INFO polygraphs> Sim #0005:     93 steps    0.02s; action: B undefined: 0 converged: 1 polarized: 0
Bye.
```

## Testing

```bash
python test_ptgraph.py --standalone
```

## Polygraphs Installation
1. Clone the Polygraphs repository:

``` bash
git clone https://github.com/alexandroskoliousis/polygraphs.git
```

2. `cd` into the `polygraphs` directory and change `requirements.txt` to remove DGL and install the latest version of PyTorch:
``` raw
torch
matplotlib
jupyterlab
tqdm=4.64.1
fsspec
ipywidgets
pylint
flake8
PyYaml
pandas
h5py
fsspec
```

4. If you need GPU support, [install the appropriate version of torch](https://pytorch.org/get-started/locally/) before continuing. For CPUs, continue to the next step

5. Clone ptgraph to replace DGL inside the Polygraphs repository:

```bash
git subtree add --prefix=dgl https://github.com/amil-m/ptgraph.git main --squash
```

6. Finish the installation from inside the Polygraphs directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
python setup.py install
```

## Attribution
This package contains code derived from [DGL](https://github.com/dmlc/dgl), licensed under the Apache License 2.0
