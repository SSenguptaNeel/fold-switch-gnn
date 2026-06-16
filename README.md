# fold-switch-gnn

A graph neural network classifier for metamorphic (fold-switching) proteins,
built with PyTorch Geometric and BioPython.

## What this does

Some proteins adopt two structurally distinct folds depending on their
environment—these are called **metamorphic** or **fold-switching** proteins.
XCL1/lymphotactin is the canonical example: it interconverts between a
disordered monomeric helical form and a canonical chemokine β-sheet fold.

This project frames fold-switch detection as a **binary graph classification**
task: given a protein structure as a residue-contact graph, predict whether
it is metamorphic (label=1) or conformationally stable (label=0).

## Project structure

```
fold_switch_gnn/
├── data/
│   └── protein_list.py      # curated dataset: known switchers + stable controls
├── graphs/
│   ├── graph_builder.py     # PDB structure → PyG Data object
│   └── dataset.py           # InMemoryDataset with PDB downloading + caching
├── models/
│   └── model.py             # GCN classifier
├── train.py                 # k-fold cross-validation training
└── test_graph_builder.py    # unit tests (no network needed)
```

## Graph representation

Each protein chain is encoded as an undirected graph:

**Nodes** — one per residue (Cα as position)

| Feature slice | Dimension | Description |
|---|---|---|
| `x[:, 0:20]` | 20 | One-hot amino acid identity |
| `x[:, 20]` | 1 | Relative solvent accessibility (DSSP, optional) |
| `x[:, 21:25]` | 4 | Backbone dihedrals: sin/cos of φ, ψ |
| `x[:, 25]` | 1 | Normalised sequence position |

**Edges** — two types:
- **Sequential bonds**: always added between residues i and i+1
- **Spatial contacts**: Cα–Cα distance ≤ 8 Å

**Edge features** (3-dim): normalised distance, log-normalised sequence
separation, bond type indicator (sequential vs spatial)

## Model

Three-layer GCN with mean+max pooling readout and an MLP head.
Trained with class-weighted BCE loss to handle the imbalanced dataset.

```
Input (26-dim) → GCNConv → BN → ReLU  (×3)
               → global_mean_pool ⊕ global_max_pool
               → Linear(128, 64) → ReLU → Dropout → Linear(64, 1)
```

## Usage

```bash
# Install dependencies
pip install torch torch-geometric biopython scikit-learn

# Run unit tests (no internet needed)
python -m fold_switch_gnn.test_graph_builder

# Build dataset and train (downloads PDB files from RCSB)
python -m fold_switch_gnn.train --root ./cache --epochs 100 --folds 5
```

## Dataset

The curated set includes experimentally confirmed fold-switching proteins:

| Protein | PDB IDs | Biological role |
|---|---|---|
| XCL1 / Lymphotactin | 1J9O, 2JP1 | Chemokine; helix ↔ β-sheet |
| RfaH NTD | 2LCL, 6ZJH | Transcription factor; helix-hairpin ↔ β-barrel |
| KaiB | 5JYT, 5C5E | Circadian clock; ground ↔ fold-switch state |
| CLIC1 | 1K0M | Chloride channel; soluble ↔ membrane form |
| Mad2 | 1DUJ, 1KLQ | Spindle checkpoint; open ↔ closed |

Stable controls from the CATH S40 non-redundant set (ubiquitin, villin HP36,
CI2, trp cage, etc.).

## Extending this project

- **ESM2 embeddings as node features**: replace one-hot AA identity with
  per-residue ESM2 embeddings (1280-dim) for a strong baseline comparison.
- **Edge-aware message passing**: swap `GCNConv` for `GATv2Conv` or `NNConv`
  to incorporate edge features (distance, bond type) in the aggregation.
- **Larger dataset**: DisMeta and MetSite databases contain hundreds of
  additional annotated fold-switching residues.
- **Visualisation**: use `networkx` + `py3Dmol` to render the contact graph
  overlaid on the 3D structure.

## References

- Dishman et al. (2021). *Atomic resolution dynamics of metamorphic proteins*. Science.
- Porter & Bhatt (2020). *Extant fold-switching proteins are widespread*. PLOS Comput. Biol.
- Schuler et al. (2020). *Dynamics of XCL1 metamorphosis*. PNAS.
