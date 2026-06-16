"""
dataset.py
==========
InMemoryDataset subclass that:
  1. Downloads PDB files from RCSB (or loads from cache)
  2. Parses with BioPython
  3. Calls graph_builder.structure_to_graph()
  4. Stores processed PyG Data objects

Usage
-----
    from fold_switch_gnn.graphs.dataset import FoldSwitchDataset

    dataset = FoldSwitchDataset(root="./cache")
    print(dataset[0])           # Data(x=[71, 26], edge_index=[2, 312], ...)
    print(dataset.num_features) # 26
    print(dataset.num_classes)  # 2
"""

import os
import urllib.request
from pathlib import Path

import torch
from torch_geometric.data import InMemoryDataset

from Bio.PDB import PDBParser

from fold_switch_gnn.data.protein_list import ALL_PROTEINS
from fold_switch_gnn.graphs.graph_builder import structure_to_graph


RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


def download_pdb(pdb_id: str, dest_dir: Path) -> Path:
    """Download a PDB file from RCSB if not already cached."""
    dest = dest_dir / f"{pdb_id.upper()}.pdb"
    if dest.exists():
        return dest
    url = RCSB_URL.format(pdb_id=pdb_id.upper())
    print(f"  Downloading {pdb_id} from RCSB...")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        raise RuntimeError(f"Failed to download {pdb_id}: {e}")
    return dest


class FoldSwitchDataset(InMemoryDataset):
    """
    Binary classification dataset:
      - class 1: metamorphic / fold-switching proteins
      - class 0: conformationally stable single-fold proteins

    Parameters
    ----------
    root     : directory used for raw PDB files and processed .pt cache
    proteins : list of (pdb_id, chain_id, label, name) tuples.
               Defaults to ALL_PROTEINS from data/protein_list.py
    cutoff   : Cα–Cα spatial contact cutoff in Ångström (default 8.0)
    transform, pre_transform: standard PyG hooks
    """

    def __init__(
        self,
        root: str = "./cache",
        proteins: list = None,
        cutoff: float = 8.0,
        transform=None,
        pre_transform=None,
    ):
        self.proteins = proteins if proteins is not None else ALL_PROTEINS
        self.cutoff = cutoff
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    # ── PyG Dataset interface ──────────────────────────────────────────────────

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, "raw_pdb")

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, "processed")

    @property
    def raw_file_names(self):
        return [f"{pdb_id.upper()}.pdb" for pdb_id, _, _, _ in self.proteins]

    @property
    def processed_file_names(self):
        return [f"fold_switch_dataset_cutoff{int(self.cutoff)}.pt"]

    def download(self):
        raw_path = Path(self.raw_dir)
        raw_path.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {len(self.proteins)} PDB structures...")
        for pdb_id, chain_id, label, name in self.proteins:
            try:
                download_pdb(pdb_id, raw_path)
            except RuntimeError as e:
                print(f"  WARNING: {e} — skipping {name}")

    def process(self):
        parser = PDBParser(QUIET=True)
        data_list = []
        skipped = []

        print(f"Building graphs (cutoff={self.cutoff} Å)...")

        for pdb_id, chain_id, label, name in self.proteins:
            pdb_file = Path(self.raw_dir) / f"{pdb_id.upper()}.pdb"

            if not pdb_file.exists():
                print(f"  SKIP {name} ({pdb_id}): PDB not found")
                skipped.append(name)
                continue

            try:
                structure = parser.get_structure(pdb_id, str(pdb_file))
                graph = structure_to_graph(
                    structure,
                    chain_id=chain_id,
                    label=label,
                    pdb_id=f"{pdb_id}_{name}",
                    cutoff=self.cutoff,
                )
                data_list.append(graph)
                print(f"  OK  {name:25s}  nodes={graph.num_nodes:4d}  edges={graph.edge_index.shape[1]:5d}  label={label}")

            except Exception as e:
                print(f"  ERR {name} ({pdb_id}): {e}")
                skipped.append(name)

        if skipped:
            print(f"\nSkipped {len(skipped)} proteins: {skipped}")

        print(f"\nDataset: {len(data_list)} graphs "
              f"({sum(d.y.item()==1 for d in data_list)} fold-switchers, "
              f"{sum(d.y.item()==0 for d in data_list)} stable)")

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    # ── convenience properties ─────────────────────────────────────────────────

    @property
    def num_node_features(self) -> int:
        return self[0].x.shape[1]   # 26

    @property
    def num_classes(self) -> int:
        return 2

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for imbalanced classes."""
        labels = torch.tensor([self[i].y.item() for i in range(len(self))])
        counts = torch.bincount(labels, minlength=2).float()
        weights = 1.0 / (counts + 1e-6)
        return weights / weights.sum()

    def summary(self):
        """Print a human-readable dataset summary."""
        n = len(self)
        n1 = sum(self[i].y.item() == 1 for i in range(n))
        n0 = n - n1
        avg_nodes = sum(self[i].num_nodes for i in range(n)) / n
        avg_edges = sum(self[i].edge_index.shape[1] for i in range(n)) / n
        print(f"FoldSwitchDataset")
        print(f"  Total proteins : {n}")
        print(f"  Fold-switchers : {n1}  (label=1)")
        print(f"  Stable         : {n0}  (label=0)")
        print(f"  Avg nodes      : {avg_nodes:.1f}")
        print(f"  Avg edges      : {avg_edges:.1f}")
        print(f"  Node feat dim  : {self.num_node_features}")
        print(f"  Edge feat dim  : {self[0].edge_attr.shape[1]}")
