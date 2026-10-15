import numpy as np
import simulate


def test_moisture_declines_with_no_rain_no_irrigation():
    m0 = 180.0
    m1 = simulate.step_moisture(m0, rainfall_mm=0, et0_mm=5, water_demand_coefficient=0.9,
                                 root_capacity_mm=180, drainage_rate=0.15)
    assert m1 < m0


def test_moisture_never_goes_negative_under_extreme_dryness():
    m = 5.0
    for _ in range(30):
        m = simulate.step_moisture(m, rainfall_mm=0, et0_mm=8, water_demand_coefficient=1.0,
                                    root_capacity_mm=180, drainage_rate=0.15)
    assert m >= 0.0


def test_heavy_rain_can_temporarily_exceed_capacity_but_stays_bounded():
    m = 180.0
    m = simulate.step_moisture(m, rainfall_mm=200, et0_mm=3, water_demand_coefficient=0.6,
                                root_capacity_mm=180, drainage_rate=0.1)
    assert m > 180.0
    assert m <= 180.0 * simulate.MAX_SATURATION_FACTOR


def test_excess_moisture_drains_back_toward_capacity_over_time():
    m = 180.0 * simulate.MAX_SATURATION_FACTOR
    for _ in range(10):
        m = simulate.step_moisture(m, rainfall_mm=0, et0_mm=3, water_demand_coefficient=0.6,
                                    root_capacity_mm=180, drainage_rate=0.3)
    assert m < 180.0 * simulate.MAX_SATURATION_FACTOR
    assert m >= 0


def test_irrigation_adds_water():
    m0 = 100.0
    m_no_irrigation = simulate.step_moisture(m0, rainfall_mm=0, et0_mm=4, water_demand_coefficient=0.8,
                                              root_capacity_mm=180, drainage_rate=0.15, irrigation_mm=0)
    m_with_irrigation = simulate.step_moisture(m0, rainfall_mm=0, et0_mm=4, water_demand_coefficient=0.8,
                                               root_capacity_mm=180, drainage_rate=0.15, irrigation_mm=40)
    assert m_with_irrigation > m_no_irrigation


def test_water_stress_index_at_full_capacity_is_zero():
    assert simulate.water_stress_index(moisture=180, root_capacity_mm=180, stress_sensitivity=1.0) == 0.0


def test_water_stress_index_at_zero_moisture_is_one():
    assert simulate.water_stress_index(moisture=0, root_capacity_mm=180, stress_sensitivity=1.0) == 1.0


def test_water_stress_index_is_bounded_even_with_high_sensitivity():
    stress = simulate.water_stress_index(moisture=10, root_capacity_mm=180, stress_sensitivity=5.0)
    assert 0.0 <= stress <= 1.0


def test_young_trees_show_more_stress_at_same_moisture():
    mature_stress = simulate.water_stress_index(moisture=100, root_capacity_mm=180, stress_sensitivity=1.0)
    young_stress = simulate.water_stress_index(moisture=100, root_capacity_mm=180, stress_sensitivity=1.3)
    assert young_stress > mature_stress


def test_observed_moisture_is_bounded():
    rng = np.random.default_rng(0)
    for _ in range(200):
        obs = simulate.observed_moisture(true_moisture=170, root_capacity_mm=180, rng=rng)
        assert 0.0 <= obs <= 180 * simulate.MAX_SATURATION_FACTOR


def test_visible_stress_rating_is_in_valid_range():
    rng = np.random.default_rng(0)
    for _ in range(200):
        rating = simulate.visible_stress_rating(stress_today=0.5, stress_yesterday=0.4, rng=rng)
        assert rating in (0, 1, 2, 3)
