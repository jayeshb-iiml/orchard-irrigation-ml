import policy


def test_low_stress_is_none():
    assert policy.determine_irrigation_need(stress_index=0.1, recent_rainfall_mm=0) == policy.NONE


def test_mid_stress_is_monitor():
    assert policy.determine_irrigation_need(stress_index=0.45, recent_rainfall_mm=0) == policy.MONITOR


def test_high_stress_is_irrigate():
    assert policy.determine_irrigation_need(stress_index=0.75, recent_rainfall_mm=0) == policy.IRRIGATE


def test_boundary_values_use_the_upper_bucket():
    cfg = policy.DEFAULT_CONFIG
    assert policy.determine_irrigation_need(cfg["monitor_stress_threshold"], 0) == policy.MONITOR
    assert policy.determine_irrigation_need(cfg["irrigate_stress_threshold"], 0) == policy.IRRIGATE


def test_recent_rain_downgrades_irrigate_to_monitor():
    cfg = policy.DEFAULT_CONFIG
    high_stress = cfg["irrigate_stress_threshold"] + 0.1
    assert policy.determine_irrigation_need(high_stress, recent_rainfall_mm=0) == policy.IRRIGATE
    assert policy.determine_irrigation_need(
        high_stress, recent_rainfall_mm=cfg["recent_rain_override_mm"]
    ) == policy.MONITOR


def test_recent_rain_does_not_affect_already_low_stress():
    assert policy.determine_irrigation_need(stress_index=0.1, recent_rainfall_mm=50) == policy.NONE


def test_irrigation_amount_zero_when_already_at_target():
    cfg = policy.DEFAULT_CONFIG
    target = 180 * cfg["irrigation_target_fill_fraction"]
    assert policy.irrigation_amount_mm(target, 180) == 0.0


def test_irrigation_amount_positive_when_dry():
    amount = policy.irrigation_amount_mm(true_moisture=50, root_capacity_mm=180)
    assert amount > 0


def test_irrigation_amount_never_negative_when_already_above_target():
    amount = policy.irrigation_amount_mm(true_moisture=175, root_capacity_mm=180)
    assert amount == 0.0
