"""
test_graph_builder.py
=====================
Offline unit tests for graph_builder.structure_to_graph().
Builds a tiny synthetic structure so no network/PDB download is needed.
"""

import sys
import numpy as np
import torch

sys.path.insert(0, "/home/claude")

from fold_switch_gnn.graphs.graph_builder import (
    structure_to_graph,
    residue_to_onehot,
    calc_dihedrals,
    AA_TO_IDX,
)

# ── synthetic BioPython-like stubs ────────────────────────────────────────────

class FakeAtom:
    def __init__(self, coord):
        self._coord = np.array(coord, dtype=float)
    def get_coord(self):
        return self._coord
    def get_vector(self):
        from Bio.PDB.vectors import Vector
        return Vector(self._coord)

class FakeResidue:
    def __init__(self, resname, atoms):
        self.resname = resname
        self.atoms = atoms
        self._id = (" ", 1, " ")
    def get_resname(self): return self.resname
    def get_id(self):      return self._id
    def __contains__(self, name): return name in self.atoms
    def __getitem__(self, name):  return self.atoms[name]

class FakeChain:
    def __init__(self, residues):
        self._residues = residues
    def get_residues(self): return iter(self._residues)

class FakeModel:
    def __init__(self, chain):
        self._chain = chain
    def __getitem__(self, chain_id): return self._chain

class FakeStructure:
    def __init__(self, model):
        self._model = model
    def __getitem__(self, idx): return self._model


def make_helix(n=10):
    """Create n residues arranged as an idealised alpha helix."""
    residues = []
    aa_cycle = ["ALA", "LEU", "GLU", "LYS", "ALA", "LEU", "GLU", "LYS", "ALA", "LEU"]
    for i in range(n):
        # rough helix parameters: rise 1.5 Å, rotation 100° per residue
        angle = np.radians(i * 100)
        x = 2.3 * np.cos(angle)
        y = 2.3 * np.sin(angle)
        z = 1.5 * i
        atoms = {
            "CA": FakeAtom([x, y, z]),
            "N":  FakeAtom([x - 1.0, y, z - 0.5]),
            "C":  FakeAtom([x + 1.0, y, z + 0.5]),
        }
        residues.append(FakeResidue(aa_cycle[i % len(aa_cycle)], atoms))
    return residues


def test_onehot():
    v = residue_to_onehot("ALA")
    assert v.shape == (20,)
    assert v[AA_TO_IDX["A"]] == 1.0
    assert v.sum() == 1.0

    v_unk = residue_to_onehot("UNK")
    assert v_unk.sum() == 0.0
    print("  ✓ one-hot encoding")


def test_graph_shape():
    residues = make_helix(n=10)
    chain    = FakeChain(residues)
    model    = FakeModel(chain)
    struct   = FakeStructure(model)

    graph = structure_to_graph(struct, chain_id="A", label=1, pdb_id="FAKE", cutoff=8.0)

    assert graph.x.shape[0] == 10,   f"expected 10 nodes, got {graph.x.shape[0]}"
    assert graph.x.shape[1] == 26,   f"expected 26 features, got {graph.x.shape[1]}"
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_attr.shape[1]  == 3
    assert graph.y.item() == 1
    assert graph.num_nodes == 10
    print(f"  ✓ graph shape: x={graph.x.shape}, edges={graph.edge_index.shape[1]}, edge_attr={graph.edge_attr.shape}")


def test_edges_symmetric():
    residues = make_helix(n=8)
    chain    = FakeChain(residues)
    struct   = FakeStructure(FakeModel(chain))
    graph    = structure_to_graph(struct, chain_id="A", label=0, pdb_id="SYM")

    src, dst = graph.edge_index
    # every (i→j) should have a matching (j→i)
    edges = set(zip(src.tolist(), dst.tolist()))
    for i, j in edges:
        assert (j, i) in edges, f"missing reverse edge ({j},{i})"
    print(f"  ✓ edge symmetry ({len(edges)} directed edges)")


def test_sequential_edges_present():
    """All consecutive residue pairs must have an edge regardless of cutoff."""
    residues = make_helix(n=6)
    chain    = FakeChain(residues)
    struct   = FakeStructure(FakeModel(chain))
    graph    = structure_to_graph(struct, chain_id="A", label=0, cutoff=0.1)  # tiny cutoff

    src, dst = graph.edge_index
    edges = set(zip(src.tolist(), dst.tolist()))
    for i in range(5):
        assert (i, i+1) in edges, f"missing sequential edge ({i},{i+1})"
    print("  ✓ sequential edges always present")


def test_feature_ranges():
    residues = make_helix(n=8)
    chain    = FakeChain(residues)
    struct   = FakeStructure(FakeModel(chain))
    graph    = structure_to_graph(struct, chain_id="A", label=1)

    x = graph.x
    # one-hot part should be 0/1
    assert x[:, :20].min() >= 0.0 and x[:, :20].max() <= 1.0
    # dihedral sin/cos bounded [-1, 1]
    assert x[:, 21:25].abs().max() <= 1.0 + 1e-5
    # normalised seq position [0, 1]
    assert x[:, 25].min() >= 0.0 and x[:, 25].max() <= 1.0 + 1e-5
    # edge distances normalised [0, 1]
    assert graph.edge_attr[:, 0].min() >= 0.0
    assert graph.edge_attr[:, 0].max() <= 1.0 + 1e-5
    print("  ✓ all feature values in expected ranges")


if __name__ == "__main__":
    print("Running graph builder tests...\n")
    test_onehot()
    test_graph_shape()
    test_edges_symmetric()
    test_sequential_edges_present()
    test_feature_ranges()
    print("\nAll tests passed ✓")
