"""
graph_builder.py
================
Converts a parsed BioPython Structure into a PyTorch Geometric Data object.

Graph definition
----------------
Nodes : one per residue (Cα atom as representative position)
Edges : spatial contacts within a Euclidean cutoff (default 8 Å) +
        sequential bonds between adjacent residues in sequence

Node features (per residue)
----------------------------
  [0:20]  one-hot amino acid identity         (20-dim)
  [20]    relative solvent accessibility      (1-dim, from DSSP if available, else 0)
  [21:25] backbone dihedral angles sin/cos    (4-dim: sin_phi, cos_phi, sin_psi, cos_psi)
  [25]    residue depth (normalised seq pos)  (1-dim)
  -------
  total: 26-dim

Edge features (per contact)
----------------------------
  [0]     Euclidean distance (Å, normalised by cutoff)
  [1]     |i - j| sequence separation (log-normalised)
  [2]     1 if sequential bond, 0 if spatial contact  (bond type)
  -------
  total: 3-dim
"""

import numpy as np
import torch
from torch_geometric.data import Data
from Bio.PDB import DSSP
import warnings

# ── amino acid vocabulary ──────────────────────────────────────────────────────

AA_SINGLE = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_SINGLE)}

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

CUTOFF_ANGSTROM = 8.0   # Cα–Cα distance threshold for spatial edges


# ── node feature helpers ───────────────────────────────────────────────────────

def residue_to_onehot(resname: str) -> np.ndarray:
    """20-dim one-hot for standard amino acids; all-zero for non-standard."""
    vec = np.zeros(20, dtype=np.float32)
    aa = THREE_TO_ONE.get(resname.upper())
    if aa and aa in AA_TO_IDX:
        vec[AA_TO_IDX[aa]] = 1.0
    return vec


def calc_dihedrals(residues: list) -> np.ndarray:
    """
    Backbone phi/psi for each residue.
    Returns (N, 4) array of [sin_phi, cos_phi, sin_psi, cos_psi].
    Terminal residues that lack a neighbour get 0.
    """
    n = len(residues)
    angles = np.zeros((n, 4), dtype=np.float32)

    def get_atom(res, name):
        try:
            return res[name].get_vector()
        except KeyError:
            return None

    for i in range(n):
        # phi: C(i-1) - N(i) - CA(i) - C(i)
        if i > 0:
            c_prev = get_atom(residues[i - 1], "C")
            n_i    = get_atom(residues[i],     "N")
            ca_i   = get_atom(residues[i],     "CA")
            c_i    = get_atom(residues[i],     "C")
            if all(v is not None for v in [c_prev, n_i, ca_i, c_i]):
                from Bio.PDB.vectors import calc_dihedral
                phi = calc_dihedral(c_prev, n_i, ca_i, c_i)
                angles[i, 0] = np.sin(phi)
                angles[i, 1] = np.cos(phi)

        # psi: N(i) - CA(i) - C(i) - N(i+1)
        if i < n - 1:
            n_i    = get_atom(residues[i],     "N")
            ca_i   = get_atom(residues[i],     "CA")
            c_i    = get_atom(residues[i],     "C")
            n_next = get_atom(residues[i + 1], "N")
            if all(v is not None for v in [n_i, ca_i, c_i, n_next]):
                from Bio.PDB.vectors import calc_dihedral
                psi = calc_dihedral(n_i, ca_i, c_i, n_next)
                angles[i, 2] = np.sin(psi)
                angles[i, 3] = np.cos(psi)

    return angles


# ── graph builder ──────────────────────────────────────────────────────────────

def structure_to_graph(
    structure,
    chain_id: str,
    label: int,
    pdb_id: str = "",
    cutoff: float = CUTOFF_ANGSTROM,
    use_dssp: bool = False,           # set True if mkdssp is installed
) -> Data:
    """
    Convert a BioPython Structure → PyTorch Geometric Data.

    Parameters
    ----------
    structure  : Bio.PDB.Structure object (already parsed)
    chain_id   : which chain to use, e.g. "A"
    label      : 1 = fold-switching, 0 = stable
    pdb_id     : string identifier for bookkeeping
    cutoff     : Cα–Cα spatial contact distance in Ångström
    use_dssp   : whether to run DSSP for RSA features

    Returns
    -------
    torch_geometric.data.Data with fields:
        x          : (N, 26)  node feature matrix
        edge_index : (2, E)   COO-format edge list
        edge_attr  : (E, 3)   edge feature matrix
        y          : (1,)     binary label
        num_nodes  : int
        pdb_id     : str
        seq        : str      one-letter sequence
    """
    model = structure[0]           # first MODEL entry in PDB

    # collect standard residues with Cα
    chain = model[chain_id]
    residues = [
        res for res in chain.get_residues()
        if res.get_id()[0] == " "               # exclude HETATM, waters
        and "CA" in res
    ]

    if len(residues) < 5:
        raise ValueError(f"{pdb_id}: chain {chain_id} has too few residues ({len(residues)})")

    n = len(residues)

    # ── Cα coordinates ────────────────────────────────────────────────────────
    ca_coords = np.array([res["CA"].get_coord() for res in residues], dtype=np.float32)

    # ── DSSP (optional) ───────────────────────────────────────────────────────
    rsa_values = np.zeros(n, dtype=np.float32)
    if use_dssp:
        try:
            dssp = DSSP(model, structure.header.get("idcode", pdb_id) + ".pdb")
            for i, res in enumerate(residues):
                key = (chain_id, res.get_id())
                if key in dssp:
                    rsa_values[i] = dssp[key][3] or 0.0   # relative ASA
        except Exception:
            warnings.warn(f"DSSP failed for {pdb_id}, RSA set to 0")

    # ── dihedral angles ───────────────────────────────────────────────────────
    dihedrals = calc_dihedrals(residues)   # (N, 4)

    # ── normalised sequence position ──────────────────────────────────────────
    seq_pos = np.linspace(0.0, 1.0, n, dtype=np.float32).reshape(-1, 1)  # (N, 1)

    # ── one-hot amino acid ────────────────────────────────────────────────────
    onehot = np.array([residue_to_onehot(res.get_resname()) for res in residues])  # (N, 20)

    # ── assemble node features ────────────────────────────────────────────────
    #   [onehot(20) | rsa(1) | dihedrals(4) | seq_pos(1)] = 26-dim
    x = np.concatenate(
        [onehot, rsa_values.reshape(-1, 1), dihedrals, seq_pos],
        axis=1
    )                                                                      # (N, 26)

    # ── build edges ───────────────────────────────────────────────────────────
    src_list, dst_list, edge_feats = [], [], []

    # pairwise Cα distances
    diff = ca_coords[:, None, :] - ca_coords[None, :, :]    # (N, N, 3)
    dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))         # (N, N)

    for i in range(n):
        for j in range(i + 1, n):
            d = dist_matrix[i, j]
            is_sequential = (j == i + 1)

            # include if: sequential neighbour OR within spatial cutoff
            if is_sequential or d <= cutoff:
                seq_sep = abs(i - j)
                log_sep = np.log1p(seq_sep) / np.log1p(n)    # normalise
                bond_type = 1.0 if is_sequential else 0.0
                d_norm = min(d / cutoff, 1.0)

                feat = [d_norm, log_sep, bond_type]

                # undirected → add both directions
                src_list += [i, j]
                dst_list += [j, i]
                edge_feats += [feat, feat]

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(edge_feats,            dtype=torch.float32)
    x_tensor   = torch.tensor(x,                     dtype=torch.float32)
    y_tensor   = torch.tensor([label],               dtype=torch.long)

    seq = "".join(
        THREE_TO_ONE.get(res.get_resname().upper(), "X")
        for res in residues
    )

    return Data(
        x=x_tensor,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y_tensor,
        num_nodes=n,
        pdb_id=pdb_id,
        seq=seq,
    )
