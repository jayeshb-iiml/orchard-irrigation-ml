"""The deterministic ground-truth irrigation policy: zero ML, fully
auditable, the same spirit as autonomy.py in the agent-autonomy-journey
repo. It consumes the simulator's TRUE state (not the noisy observed
version features.py builds for the model) and decides a 3-class label.

This encodes, in explicit code, the factors described as driving the real
decision on the orchard: soil dampness (the stress index), how long it's
been since the block was last watered (already folded into the stress
index, since moisture keeps falling the longer it goes unwatered), the
season (folded in via real weather driving evapotranspiration and
rainfall, not a hardcoded month rule — a hardcoded month would let the
model shortcut to "what month is it" instead of learning from conditions),
and whether it has rained recently (the explicit override below).

The thresholds are illustrative modeling assumptions, not validated
agronomic science, and are deliberately kept in one place so they can be
inspected, challenged, and changed without touching anything else.
"""

NONE = 0
MONITOR = 1
IRRIGATE = 2

LABELS = {
    NONE: "No irrigation",
    MONITOR: "Monitor",
    IRRIGATE: "Irrigate",
}

DEFAULT_CONFIG = {
    "monitor_stress_threshold": 0.35,
    "irrigate_stress_threshold": 0.60,
    # If it has rained at least this much in the last 2 days, downgrade an
    # "irrigate" call to "monitor" — ground is visibly wet, wait and see,
    # matching real judgment rather than mechanically following the bucket
    # model through a rain event it hasn't finished draining from yet.
    "recent_rain_override_mm": 10.0,
    # The irrigation event (when triggered) refills the block to this
    # fraction of its root-zone capacity. Only class 2 actually applies
    # water in the simulation; class 1 ("Monitor") means closer attention,
    # not a partial watering — there is no defensible rule here for how
    # much "light irrigation" would be, so it isn't claimed.
    "irrigation_target_fill_fraction": 0.85,
}


def determine_irrigation_need(stress_index, recent_rainfall_mm, config=None):
    """stress_index: 0-1 from simulate.water_stress_index, computed on the
    TRUE moisture state. recent_rainfall_mm: actual rainfall over the last
    2 days (not noisy — weather station data, not a soil guess).
    """
    cfg = config or DEFAULT_CONFIG

    if stress_index >= cfg["irrigate_stress_threshold"]:
        if recent_rainfall_mm >= cfg["recent_rain_override_mm"]:
            return MONITOR
        return IRRIGATE
    if stress_index >= cfg["monitor_stress_threshold"]:
        return MONITOR
    return NONE


def irrigation_amount_mm(true_moisture, root_capacity_mm, config=None):
    """How much water (mm equivalent) an "Irrigate" decision applies,
    feeding back into tomorrow's moisture state. Zero for any other label.
    """
    cfg = config or DEFAULT_CONFIG
    target = root_capacity_mm * cfg["irrigation_target_fill_fraction"]
    return max(0.0, target - true_moisture)
