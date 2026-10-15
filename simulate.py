"""The orchard state simulator: a daily soil-moisture water balance per
block, plus the (noisy, lagged) plant-stress signals a person would actually
observe. This file knows nothing about irrigation *decisions* — it only
advances physical state given whatever rainfall, weather, and irrigation
actually happened that day. The decision of whether to irrigate lives in
policy.py; generate_dataset.py wires the two together day by day.

The water balance is a simple single-layer "bucket" model, standard in
introductory agronomy: moisture rises with rainfall and irrigation, falls
with crop evapotranspiration (ET0 from real weather data x a species
water-demand coefficient, the FAO-56 Kc approach), and drains when it
exceeds the block's root-zone capacity. It is a simplification — no runoff,
no multi-layer soil, no lateral flow between blocks — and that's stated
plainly in the README, not hidden.
"""

import numpy as np

MAX_SATURATION_FACTOR = 1.3  # allows temporary saturation above capacity after heavy rain


def step_moisture(prev_moisture, rainfall_mm, et0_mm, water_demand_coefficient,
                   root_capacity_mm, drainage_rate, irrigation_mm=0.0):
    """Advance soil moisture by one day for one block. Pure function, no
    randomness, fully deterministic given its inputs — this is the part a
    unit test can pin down exactly.
    """
    moisture = prev_moisture + rainfall_mm + irrigation_mm

    etc_mm = et0_mm * water_demand_coefficient
    actual_et = min(etc_mm, moisture)
    moisture -= actual_et

    excess = max(0.0, moisture - root_capacity_mm)
    drainage_loss = excess * drainage_rate
    moisture -= drainage_loss

    moisture = max(0.0, min(moisture, root_capacity_mm * MAX_SATURATION_FACTOR))
    return moisture


def water_stress_index(moisture, root_capacity_mm, stress_sensitivity):
    """0 (no stress, moisture at or above capacity) to 1 (fully depleted).
    Scaled by the block's age-driven stress sensitivity — young trees show
    more stress for the same relative moisture deficit.
    """
    fill_fraction = min(1.0, moisture / root_capacity_mm)
    raw_stress = 1.0 - fill_fraction
    return float(np.clip(raw_stress * stress_sensitivity, 0.0, 1.0))


def visible_stress_rating(stress_today, stress_yesterday, rng, noise_sd=0.12):
    """A person's visual assessment: lags the true state (symptoms don't
    appear instantly), carries observation noise, and is discretized into
    the 0-3 scale a human would actually report. This is deliberately
    imperfect — it is meant to approximate judgment, not measure anything.
    """
    smoothed = 0.7 * stress_today + 0.3 * stress_yesterday
    noisy = smoothed + rng.normal(0, noise_sd)
    noisy = float(np.clip(noisy, 0.0, 1.0))
    if noisy < 0.25:
        return 0
    if noisy < 0.5:
        return 1
    if noisy < 0.75:
        return 2
    return 3


def observed_moisture(true_moisture, root_capacity_mm, rng, noise_fraction=0.05):
    """A hand-check of soil moisture is an estimate, not a lab measurement.
    Noise scales with the block's capacity so wetter-capacity blocks don't
    get unrealistically precise readings.
    """
    noise_sd = root_capacity_mm * noise_fraction
    noisy = true_moisture + rng.normal(0, noise_sd)
    return float(np.clip(noisy, 0.0, root_capacity_mm * MAX_SATURATION_FACTOR))
