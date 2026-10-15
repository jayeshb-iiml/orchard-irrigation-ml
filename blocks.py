"""The orchard's block layout: static characteristics assigned once, not
simulated. Grouping ~1,600 real trees into 12 blocks by species, age cohort,
and soil drainage profile — matching how the orchard is actually managed
(walked and assessed block by block, not tree by tree).

Species water-demand coefficients and root-zone capacities below are
illustrative horticultural approximations, not validated agronomic data.
They encode ordering that is true in general (coconut and mango are heavy,
deep-rooted feeders; mahogany/imli/neem are comparatively self-sufficient
once established) without claiming precision neither of us can verify.
"""

SOIL_PROFILES = {
    "high_retention": {"drainage_rate": 0.08},
    "medium": {"drainage_rate": 0.15},
    "fast_draining": {"drainage_rate": 0.28},
}

AGE_COHORTS = {
    "young": {"root_capacity_fraction": 0.55, "stress_sensitivity": 1.3},
    "mature": {"root_capacity_fraction": 1.0, "stress_sensitivity": 1.0},
}

SPECIES_PARAMS = {
    "coconut": {"water_demand_coefficient": 0.95, "root_capacity_mm": 180},
    "mango": {"water_demand_coefficient": 0.85, "root_capacity_mm": 150},
    "chickoo": {"water_demand_coefficient": 0.70, "root_capacity_mm": 110},
    "custard_apple": {"water_demand_coefficient": 0.60, "root_capacity_mm": 90},
    "mahogany": {"water_demand_coefficient": 0.40, "root_capacity_mm": 160},
    "imli": {"water_demand_coefficient": 0.30, "root_capacity_mm": 170},
    "neem": {"water_demand_coefficient": 0.30, "root_capacity_mm": 160},
}

# Non-commercial species: not harvested, but included so the model and the
# dashboard both have to handle a heterogeneous, low-priority block sensibly
# rather than only ever seeing high-value species.
NON_COMMERCIAL_SPECIES = {"mahogany", "imli", "neem"}

BLOCKS = [
    {"id": "C1", "species": "coconut", "tree_count": 200, "age_cohort": "mature", "soil_profile": "medium"},
    {"id": "C2", "species": "coconut", "tree_count": 150, "age_cohort": "mature", "soil_profile": "fast_draining"},
    {"id": "C3", "species": "coconut", "tree_count": 150, "age_cohort": "young", "soil_profile": "high_retention"},
    {"id": "M1", "species": "mango", "tree_count": 250, "age_cohort": "mature", "soil_profile": "medium"},
    {"id": "M2", "species": "mango", "tree_count": 150, "age_cohort": "mature", "soil_profile": "fast_draining"},
    {"id": "M3", "species": "mango", "tree_count": 100, "age_cohort": "young", "soil_profile": "high_retention"},
    {"id": "CH1", "species": "chickoo", "tree_count": 80, "age_cohort": "mature", "soil_profile": "medium"},
    {"id": "CH2", "species": "chickoo", "tree_count": 40, "age_cohort": "young", "soil_profile": "high_retention"},
    {"id": "CA1", "species": "custard_apple", "tree_count": 80, "age_cohort": "mature", "soil_profile": "fast_draining"},
    {"id": "MH1", "species": "mahogany", "tree_count": 300, "age_cohort": "mature", "soil_profile": "medium"},
    {"id": "I1", "species": "imli", "tree_count": 50, "age_cohort": "mature", "soil_profile": "fast_draining"},
    {"id": "N1", "species": "neem", "tree_count": 50, "age_cohort": "mature", "soil_profile": "fast_draining"},
]


def block_params(block):
    """Resolve a block's full static parameter set from its species, age
    cohort, and soil profile. Pure lookup, no simulation.
    """
    species = SPECIES_PARAMS[block["species"]]
    age = AGE_COHORTS[block["age_cohort"]]
    soil = SOIL_PROFILES[block["soil_profile"]]
    return {
        "water_demand_coefficient": species["water_demand_coefficient"],
        "root_capacity_mm": species["root_capacity_mm"] * age["root_capacity_fraction"],
        "stress_sensitivity": age["stress_sensitivity"],
        "drainage_rate": soil["drainage_rate"],
        "is_commercial": block["species"] not in NON_COMMERCIAL_SPECIES,
    }
