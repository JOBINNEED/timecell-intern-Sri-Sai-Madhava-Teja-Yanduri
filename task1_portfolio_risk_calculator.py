#!/usr/bin/env python3
"""
TASK 1: Portfolio Risk Calculator
Timecell.AI Summer Internship 2025 - Technical Assessment

Computes key risk metrics for portfolio safety assessment.
Includes bonus features: dual scenarios and termgraph visualizations.
"""

from termgraph.termgraph import Data, chart
from tabulate import tabulate
import numpy as np
import sys


def compute_risk_metrics(portfolio, crash_multiplier=1.0):
    """
    Compute risk metrics for a given portfolio using vectorized operations.
    
    Args:
        portfolio: Dictionary containing portfolio data
        crash_multiplier: Multiplier for crash scenario (1.0 = full crash, 0.5 = moderate)
    
    Returns:
        Dictionary with risk metrics:
        - post_crash_value: Total portfolio value after crash
        - runway_months: Months of expenses covered post-crash
        - ruin_test: 'PASS' if runway > 12 months, else 'FAIL'
        - largest_risk_asset: Asset with highest (allocation x crash magnitude)
        - concentration_warning: True if any asset > 40%
    """
    total_value = portfolio["total_value_inr"]
    monthly_expenses = portfolio["monthly_expenses_inr"]
    assets = portfolio["assets"]
    
    # Extract asset properties into parallel NumPy arrays (same index across all)
    allocations = np.array([asset["allocation_pct"] for asset in assets])
    crashes = np.array([asset["expected_crash_pct"] for asset in assets])
    names = [asset["name"] for asset in assets]
    
    # Vectorized operations
    asset_values = total_value * (allocations / 100)
    crash_impacts = crashes * crash_multiplier
    post_crash_values = asset_values * (1 + crash_impacts / 100)
    
    # Calculate total post-crash value
    post_crash_value = np.sum(post_crash_values)
    
    # Find largest risk asset using argmax (allocation × crash magnitude)
    risk_magnitudes = allocations * np.abs(crashes)
    max_risk_idx = np.argmax(risk_magnitudes)
    largest_risk_asset = names[max_risk_idx]
    
    # Check concentration warning (any asset > 40%)
    concentration_warning = np.any(allocations > 40)
    
    # Calculate runway months
    runway_months = post_crash_value / monthly_expenses if monthly_expenses > 0 else float('inf')
    
    # Ruin test
    ruin_test = "PASS" if runway_months > 12 else "FAIL"
    
    return {
        "post_crash_value": post_crash_value,
        "runway_months": runway_months,
        "ruin_test": ruin_test,
        "largest_risk_asset": largest_risk_asset,
        "concentration_warning": concentration_warning
    }


def print_allocation_chart(assets, title="Asset Allocation"):
    """Print asset allocation chart using termgraph (BONUS FEATURE)"""
    print(f"\n{title}")
    print("=" * 60)
    
    labels = [asset["name"] for asset in assets]
    data = [[asset["allocation_pct"]] for asset in assets]
    
    data_obj = Data(data, labels)
    
    args = {
        'stacked': False,
        'width': 50,
        'no_labels': False,
        'format': '{:.0f}%',
        'suffix': '',
        'vertical': False,
        'different_scale': False,
        'calendar': False,
        'start_dt': None,
        'custom_tick': '',
        'delim': '',
        'verbose': False,
        'label_before': True,
        'version': False,
        'no_values': False,
        'color': None,
        'title': None,
        'histogram': False,
    }
    
    colors = [94]
    chart(data_obj, args, colors)
    print("=" * 60)


def print_metrics(metrics, scenario_name="Crash Scenario"):
    """Print risk metrics in formatted output"""
    print(f"\n{'=' * 60}")
    print(f"  {scenario_name}")
    print(f"{'=' * 60}")
    print(f"Post-Crash Value:      ₹{metrics['post_crash_value']:,.2f}")
    print(f"Runway (Months):       {metrics['runway_months']:.1f}")
    print(f"Ruin Test:             {metrics['ruin_test']}")
    print(f"Largest Risk Asset:    {metrics['largest_risk_asset']}")
    print(f"Concentration Warning: {'⚠️  YES' if metrics['concentration_warning'] else '✓ NO'}")
    print(f"{'=' * 60}")


def main():
    """Main demonstration of portfolio risk calculator"""
    
    # Example portfolio from task specification
    portfolio = {
        "total_value_inr": 10_000_000,  # 1 Crore INR
        "monthly_expenses_inr": 80_000,
        "assets": [
            {"name": "BTC", "allocation_pct": 30, "expected_crash_pct": -80},
            {"name": "NIFTY50", "allocation_pct": 40, "expected_crash_pct": -40},
            {"name": "GOLD", "allocation_pct": 20, "expected_crash_pct": -15},
            {"name": "CASH", "allocation_pct": 10, "expected_crash_pct": 0},
        ]
    }
    
    print("\n" + "=" * 60)
    print("  TASK 1: PORTFOLIO RISK CALCULATOR")
    print("=" * 60)
    
    print(f"\nTotal Portfolio Value: ₹{portfolio['total_value_inr']:,}")
    print(f"Monthly Expenses:      ₹{portfolio['monthly_expenses_inr']:,}")
    
    # BONUS: Visualization
    print_allocation_chart(portfolio["assets"])
    
    # Core requirement: Full crash scenario
    full_crash_metrics = compute_risk_metrics(portfolio, crash_multiplier=1.0)
    print_metrics(full_crash_metrics, "Full Crash Scenario")
    
    # BONUS: Moderate crash scenario (50% of expected crash)
    moderate_crash_metrics = compute_risk_metrics(portfolio, crash_multiplier=0.5)
    print_metrics(moderate_crash_metrics, "Moderate Crash (50%)")
    
    # BONUS: Side-by-side comparison using tabulate
    print(f"\n{'=' * 60}")
    print("  SCENARIO COMPARISON")
    print(f"{'=' * 60}")
    
    comparison_data = [
        ["Post-Crash Value", 
         f"₹{full_crash_metrics['post_crash_value']:,.0f}", 
         f"₹{moderate_crash_metrics['post_crash_value']:,.0f}"],
        ["Runway (Months)", 
         f"{full_crash_metrics['runway_months']:.1f}", 
         f"{moderate_crash_metrics['runway_months']:.1f}"],
        ["Ruin Test", 
         full_crash_metrics['ruin_test'], 
         moderate_crash_metrics['ruin_test']],
        ["Largest Risk Asset",
         full_crash_metrics['largest_risk_asset'],
         moderate_crash_metrics['largest_risk_asset']],
        ["Concentration Warning",
         "⚠️  YES" if full_crash_metrics['concentration_warning'] else "✓ NO",
         "⚠️  YES" if moderate_crash_metrics['concentration_warning'] else "✓ NO"]
    ]
    
    print(tabulate(comparison_data, 
                   headers=["Metric", "Full Crash", "Moderate Crash (50%)"],
                   tablefmt="grid"))
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
