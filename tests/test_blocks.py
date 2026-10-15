from blocks import BLOCKS, block_params, SPECIES_PARAMS

EXPECTED_SPECIES_TOTALS = {
    "coconut": 500, "mango": 500, "chickoo": 120, "custard_apple": 80,
    "mahogany": 300, "imli": 50, "neem": 50,
}


def test_tree_counts_match_the_real_orchard():
    totals = {}
    for b in BLOCKS:
        totals[b["species"]] = totals.get(b["species"], 0) + b["tree_count"]
    assert totals == EXPECTED_SPECIES_TOTALS


def test_total_tree_count_is_1600():
    assert sum(b["tree_count"] for b in BLOCKS) == 1600


def test_every_block_species_has_params():
    for b in BLOCKS:
        assert b["species"] in SPECIES_PARAMS


def test_young_trees_have_smaller_root_capacity_than_mature_same_species():
    young_blocks = {b["species"]: b for b in BLOCKS if b["age_cohort"] == "young"}
    mature_blocks = {b["species"]: b for b in BLOCKS if b["age_cohort"] == "mature"}
    for species, young in young_blocks.items():
        if species in mature_blocks:
            young_cap = block_params(young)["root_capacity_mm"]
            mature_cap = block_params(mature_blocks[species])["root_capacity_mm"]
            assert young_cap < mature_cap


def test_non_commercial_species_flagged_correctly():
    for b in BLOCKS:
        params = block_params(b)
        if b["species"] in ("mahogany", "imli", "neem"):
            assert params["is_commercial"] is False
        else:
            assert params["is_commercial"] is True
