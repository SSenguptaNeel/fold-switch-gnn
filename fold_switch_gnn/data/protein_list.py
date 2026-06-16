"""
Curated protein dataset for fold-switch GNN classifier.

Fold-switching proteins: proteins experimentally confirmed to adopt two
distinct secondary-structure topologies depending on conditions.

Labels: 1 = fold-switching (metamorphic), 0 = conformationally stable

Sources:
- Fold-switchers: Porter & Bhatt (2020) PLOS Comput. Biol.; Dishman et al. (2021) Science
- Stable controls: single-fold proteins from CATH S40 non-redundant set
"""

FOLD_SWITCHERS = [
    # (pdb_id, chain, label, description)
    # XCL1 / Lymphotactin - the canonical example, your reading project!
    ("1J9O", "A", 1, "XCL1_alpha"),        # helical form
    ("2JP1", "A", 1, "XCL1_beta"),          # chemokine fold form
    # RfaH NTD - switches between helix-hairpin and beta-barrel
    ("2LCL", "A", 1, "RfaH_helix"),
    ("6ZJH", "A", 1, "RfaH_beta"),
    # KaiB - circadian clock protein, switches quaternary + fold
    ("5JYT", "A", 1, "KaiB_ground"),
    ("5C5E", "A", 1, "KaiB_fold_switch"),
    # CLIC1 - chloride channel, fold-switches on membrane interaction
    ("1K0M", "A", 1, "CLIC1_soluble"),
    # Mad2 - spindle checkpoint, O-Mad2 vs C-Mad2
    ("1DUJ", "A", 1, "Mad2_open"),
    ("1KLQ", "A", 1, "Mad2_closed"),
    # HIV-1 Rev protein ARM
    ("1ETF", "A", 1, "Rev_ARM"),
]

STABLE_CONTROLS = [
    # Well-characterised single-fold proteins, diverse structural classes
    ("1UBQ", "A", 0, "Ubiquitin"),
    ("1VII", "A", 0, "Villin_HP36"),
    ("2CI2", "A", 0, "CI2_inhibitor"),
    ("1L2Y", "A", 0, "Trp_cage"),
    ("1ENH", "A", 0, "Engrailed_homeodomain"),
    ("3GB1", "A", 0, "Protein_G_B1"),
    ("1FME", "A", 0, "FME_alpha"),
    ("2F21", "A", 0, "WW_domain"),
    ("1PRB", "A", 0, "PrB_beta"),
    ("1HZ6", "A", 0, "FSD1"),
]

ALL_PROTEINS = FOLD_SWITCHERS + STABLE_CONTROLS
