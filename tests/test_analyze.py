import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest
from main_pipeline import analyze


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "revenue":  [100.0, 200.0, -50.0, 300.0],
        "returned": [0, 0, 1, 0],
        "category": ["A", "B", "A", "B"],
        "product":  ["P1", "P2", "P1", "P3"],
        "region":   ["UK", "UK", "France", "UK"],
    })


def test_analyze_total_revenue(sample_df):
    stats = analyze(sample_df)
    assert stats["total_revenue"] == 550.0


def test_analyze_total_orders(sample_df):
    stats = analyze(sample_df)
    assert stats["total_orders"] == 4


def test_analyze_return_rate(sample_df):
    stats = analyze(sample_df)
    assert stats["return_rate"] == 25.0


def test_analyze_top_region(sample_df):
    stats = analyze(sample_df)
    assert stats["top_region"] == "UK"


def test_analyze_top_product(sample_df):
    stats = analyze(sample_df)
    assert stats["top_product"] == "P3"


def test_analyze_avg_order_value(sample_df):
    stats = analyze(sample_df)
    expected = (100.0 + 200.0 - 50.0 + 300.0) / 4
    assert stats["avg_order_value"] == round(expected, 2)


def test_analyze_returns_dict_with_all_keys(sample_df):
    stats = analyze(sample_df)
    expected_keys = {
        "total_revenue", "avg_order_value", "total_orders",
        "return_rate", "top_category", "top_product", "top_region"
    }
    assert expected_keys.issubset(stats.keys())