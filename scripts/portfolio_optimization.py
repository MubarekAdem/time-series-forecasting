"""
Portfolio Optimization using PyPortfolioOpt

This script performs portfolio optimization for TSLA, BND, and SPY using Modern Portfolio Theory.
Generates the Efficient Frontier and identifies Max Sharpe and Min Volatility portfolios.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt import plotting
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_portfolio_data():
    """Load historical data for all three assets."""
    print("Loading portfolio data...")
    
    assets = {}
    for ticker in ['TSLA', 'BND', 'SPY']:
        file_path = f'data/processed/{ticker}_processed.csv'
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        assets[ticker] = df['Close']
    
    # Combine into single DataFrame
    prices = pd.DataFrame(assets)
    prices = prices.dropna()
    
    print(f"Data range: {prices.index.min()} to {prices.index.max()}")
    print(f"Number of observations: {len(prices)}")
    
    return prices

def calculate_expected_returns_and_cov(prices):
    """Calculate expected returns and covariance matrix."""
    print("\nCalculating expected returns and covariance matrix...")
    
    # Method 1: Mean historical returns
    mu = expected_returns.mean_historical_return(prices, frequency=252)
    print("\nExpected Annual Returns:")
    for ticker, ret in mu.items():
        print(f"  {ticker}: {ret*100:.2f}%")
    
    # Covariance matrix
    S = risk_models.sample_cov(prices, frequency=252)
    print("\nCovariance Matrix:")
    print(S)
    
    return mu, S

def optimize_max_sharpe(mu, S):
    """Find the Maximum Sharpe Ratio portfolio."""
    print("\n" + "="*60)
    print("Optimizing for Maximum Sharpe Ratio...")
    print("="*60)
    
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe(risk_free_rate=0.04)  # Assuming 4% risk-free rate
    cleaned_weights = ef.clean_weights()
    
    performance = ef.portfolio_performance(risk_free_rate=0.04)
    expected_return, volatility, sharpe_ratio = performance
    
    print("\nMax Sharpe Ratio Portfolio:")
    print(f"  Expected Return: {expected_return*100:.2f}%")
    print(f"  Volatility (Risk): {volatility*100:.2f}%")
    print(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
    print("\n  Asset Allocation:")
    for ticker, weight in cleaned_weights.items():
        if weight > 0.001:  # Only show allocations > 0.1%
            print(f"    {ticker}: {weight*100:.2f}%")
    
    portfolio_data = {
        'weights': cleaned_weights,
        'expected_return': expected_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'type': 'Max Sharpe Ratio'
    }
    
    return portfolio_data

def optimize_min_volatility(mu, S):
    """Find the Minimum Volatility portfolio."""
    print("\n" + "="*60)
    print("Optimizing for Minimum Volatility...")
    print("="*60)
    
    ef = EfficientFrontier(mu, S)
    weights = ef.min_volatility()
    cleaned_weights = ef.clean_weights()
    
    performance = ef.portfolio_performance(risk_free_rate=0.04)
    expected_return, volatility, sharpe_ratio = performance
    
    print("\nMin Volatility Portfolio:")
    print(f"  Expected Return: {expected_return*100:.2f}%")
    print(f"  Volatility (Risk): {volatility*100:.2f}%")
    print(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
    print("\n  Asset Allocation:")
    for ticker, weight in cleaned_weights.items():
        if weight > 0.001:
            print(f"    {ticker}: {weight*100:.2f}%")
    
    portfolio_data = {
        'weights': cleaned_weights,
        'expected_return': expected_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'type': 'Minimum Volatility'
    }
    
    return portfolio_data

def plot_efficient_frontier(mu, S, max_sharpe_portfolio, min_vol_portfolio):
    """Plot the Efficient Frontier with marked portfolios."""
    print("\nGenerating Efficient Frontier plot...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Generate efficient frontier points
    ef = EfficientFrontier(mu, S)
    ax = plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)
    
    # Mark Max Sharpe portfolio
    max_sharpe_ret = max_sharpe_portfolio['expected_return']
    max_sharpe_vol = max_sharpe_portfolio['volatility']
    ax.scatter(max_sharpe_vol, max_sharpe_ret, marker='*', s=500, c='gold', 
               edgecolors='black', linewidths=2, label='Max Sharpe Ratio', zorder=5)
    
    # Mark Min Volatility portfolio
    min_vol_ret = min_vol_portfolio['expected_return']
    min_vol_vol = min_vol_portfolio['volatility']
    ax.scatter(min_vol_vol, min_vol_ret, marker='D', s=300, c='limegreen', 
               edgecolors='black', linewidths=2, label='Min Volatility', zorder=5)
    
    # Add annotations
    ax.annotate(f'Max Sharpe\nSR={max_sharpe_portfolio["sharpe_ratio"]:.2f}', 
                xy=(max_sharpe_vol, max_sharpe_ret), 
                xytext=(max_sharpe_vol+0.02, max_sharpe_ret+0.02),
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    
    ax.annotate(f'Min Vol\nRisk={min_vol_portfolio["volatility"]*100:.1f}%', 
                xy=(min_vol_vol, min_vol_ret), 
                xytext=(min_vol_vol-0.05, min_vol_ret-0.02),
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', lw=1.5))
    
    ax.set_title('Efficient Frontier - TSLA, BND, SPY Portfolio', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Volatility (Risk)', fontsize=13)
    ax.set_ylabel('Expected Return', fontsize=13)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = 'outputs/portfolio/efficient_frontier.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Efficient Frontier plot to {output_path}")
    plt.close()

def generate_recommendation(max_sharpe_portfolio, min_vol_portfolio, mu):
    """Generate final portfolio recommendation."""
    print("\n" + "="*60)
    print("Generating Portfolio Recommendation...")
    print("="*60)
    
    # Recommend Max Sharpe for growth-oriented investors
    # But provide context for both
    
    recommendation = {
        'recommended_portfolio': 'Max Sharpe Ratio',
        'rationale': 'Optimizes risk-adjusted returns for growth-oriented investors',
        'allocation': max_sharpe_portfolio['weights'],
        'metrics': {
            'expected_return': max_sharpe_portfolio['expected_return'],
            'volatility': max_sharpe_portfolio['volatility'],
            'sharpe_ratio': max_sharpe_portfolio['sharpe_ratio']
        },
        'alternative': {
            'name': 'Min Volatility',
            'allocation': min_vol_portfolio['weights'],
            'metrics': {
                'expected_return': min_vol_portfolio['expected_return'],
                'volatility': min_vol_portfolio['volatility'],
                'sharpe_ratio': min_vol_portfolio['sharpe_ratio']
            }
        },
        'individual_returns': {ticker: float(ret) for ticker, ret in mu.items()}
    }
    
    print("\n✅ RECOMMENDED: Max Sharpe Ratio Portfolio")
    print(f"\nAllocation:")
    for ticker, weight in max_sharpe_portfolio['weights'].items():
        if weight > 0.001:
            print(f"  {ticker}: {weight*100:.2f}%")
    
    print(f"\nExpected Performance:")
    print(f"  Annual Return: {max_sharpe_portfolio['expected_return']*100:.2f}%")
    print(f"  Annual Volatility: {max_sharpe_portfolio['volatility']*100:.2f}%")
    print(f"  Sharpe Ratio: {max_sharpe_portfolio['sharpe_ratio']:.3f}")
    
    print(f"\n📊 Alternative: Min Volatility Portfolio (for conservative investors)")
    print(f"\nAllocation:")
    for ticker, weight in min_vol_portfolio['weights'].items():
        if weight > 0.001:
            print(f"  {ticker}: {weight*100:.2f}%")
    
    print(f"\nExpected Performance:")
    print(f"  Annual Return: {min_vol_portfolio['expected_return']*100:.2f}%")
    print(f"  Annual Volatility: {min_vol_portfolio['volatility']*100:.2f}%")
    print(f"  Sharpe Ratio: {min_vol_portfolio['sharpe_ratio']:.3f}")
    
    return recommendation

def save_portfolio_outputs(max_sharpe, min_vol, recommendation):
    """Save portfolio outputs to JSON files."""
    print("\nSaving portfolio outputs...")
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return obj
    
    max_sharpe = convert_to_serializable(max_sharpe)
    min_vol = convert_to_serializable(min_vol)
    recommendation = convert_to_serializable(recommendation)
    
    with open('outputs/portfolio/max_sharpe_portfolio.json', 'w') as f:
        json.dump(max_sharpe, f, indent=2)
    print("  - outputs/portfolio/max_sharpe_portfolio.json")
    
    with open('outputs/portfolio/min_volatility_portfolio.json', 'w') as f:
        json.dump(min_vol, f, indent=2)
    print("  - outputs/portfolio/min_volatility_portfolio.json")
    
    with open('outputs/portfolio/recommended_portfolio.json', 'w') as f:
        json.dump(recommendation, f, indent=2)
    print("  - outputs/portfolio/recommended_portfolio.json")

def main():
    """Main execution function."""
    print("=" * 60)
    print("Portfolio Optimization - TSLA, BND, SPY")
    print("=" * 60)
    
    # Load data
    prices = load_portfolio_data()
    
    # Calculate expected returns and covariance
    mu, S = calculate_expected_returns_and_cov(prices)
    
    # Optimize portfolios
    max_sharpe_portfolio = optimize_max_sharpe(mu, S)
    min_vol_portfolio = optimize_min_volatility(mu, S)
    
    # Plot Efficient Frontier
    plot_efficient_frontier(mu, S, max_sharpe_portfolio, min_vol_portfolio)
    
    # Generate recommendation
    recommendation = generate_recommendation(max_sharpe_portfolio, min_vol_portfolio, mu)
    
    # Save outputs
    save_portfolio_outputs(max_sharpe_portfolio, min_vol_portfolio, recommendation)
    
    print("\n" + "=" * 60)
    print("Portfolio Optimization Complete!")
    print("=" * 60)
    print("\nOutputs:")
    print("  - outputs/portfolio/efficient_frontier.png")
    print("  - outputs/portfolio/max_sharpe_portfolio.json")
    print("  - outputs/portfolio/min_volatility_portfolio.json")
    print("  - outputs/portfolio/recommended_portfolio.json")

if __name__ == "__main__":
    main()
