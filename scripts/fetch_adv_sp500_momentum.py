#!/usr/bin/env python
"""
Fetch S&P 500 Symbols

This script fetches the current S&P 500 constituents from Wikipedia
and saves them to a file for use in momentum analysis.
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
import json
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def fetch_sp500_symbols():
    """
    Fetch current S&P 500 symbols from Wikipedia
    
    Returns:
        List of S&P 500 symbols
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    try:
        # Fetch the page
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the table
        table = soup.find('table', {'id': 'constituents'})
        
        if not table:
            print("Could not find S&P 500 table on Wikipedia")
            return None
        
        # Parse table to DataFrame
        df = pd.read_html(str(table))[0]
        
        # Extract symbols
        symbols = df['Symbol'].tolist()
        
        # Clean symbols (remove any extra characters)
        symbols = [symbol.replace('.', '-') for symbol in symbols]  # Handle BRK.B -> BRK-B
        
        # Get additional info
        companies_info = []
        for idx, row in df.iterrows():
            company_info = {
                'symbol': row['Symbol'].replace('.', '-'),
                'name': row['Security'],
                'sector': row['GICS Sector'],
                'sub_industry': row['GICS Sub-Industry'],
                'headquarters': row.get('Headquarters Location', ''),
                'date_added': row.get('Date added', ''),
                'cik': row.get('CIK', ''),
                'founded': row.get('Founded', '')
            }
            companies_info.append(company_info)
        
        return symbols, companies_info
        
    except Exception as e:
        print(f"Error fetching S&P 500 symbols: {e}")
        return None, None


def save_sp500_data(symbols, companies_info):
    """
    Save S&P 500 data to files
    
    Args:
        symbols: List of symbols
        companies_info: List of company information dictionaries
    """
    # Create data directory if it doesn't exist
    data_dir = project_root / 'data' / 'universe'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save symbols list
    symbols_file = data_dir / 'sp500_symbols.json'
    with open(symbols_file, 'w') as f:
        json.dump({
            'symbols': symbols,
            'count': len(symbols),
            'updated': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"Saved {len(symbols)} symbols to {symbols_file}")
    
    # Save detailed company info
    if companies_info:
        info_file = data_dir / 'sp500_companies.json'
        with open(info_file, 'w') as f:
            json.dump({
                'companies': companies_info,
                'count': len(companies_info),
                'updated': datetime.now().isoformat()
            }, f, indent=2)
        
        # Also save as CSV for easy viewing
        csv_file = data_dir / 'sp500_companies.csv'
        df = pd.DataFrame(companies_info)
        df.to_csv(csv_file, index=False)
        
        print(f"Saved company info to {info_file} and {csv_file}")
        
        # Show sector distribution
        sector_counts = df['sector'].value_counts()
        print("\nSector Distribution:")
        for sector, count in sector_counts.items():
            print(f"  {sector}: {count} companies")


def load_sp500_symbols():
    """
    Load S&P 500 symbols from saved file
    
    Returns:
        List of symbols or None if file doesn't exist
    """
    symbols_file = project_root / 'data' / 'universe' / 'sp500_symbols.json'
    
    if symbols_file.exists():
        with open(symbols_file, 'r') as f:
            data = json.load(f)
            return data['symbols']
    
    return None


def main():
    """Main function"""
    print("Fetching current S&P 500 constituents...")
    
    # Fetch symbols
    symbols, companies_info = fetch_sp500_symbols()
    
    if symbols:
        print(f"Successfully fetched {len(symbols)} S&P 500 symbols")
        
        # Save to files
        save_sp500_data(symbols, companies_info)
        
        # Display first 10 symbols
        print("\nFirst 10 symbols:")
        for i, symbol in enumerate(symbols[:10], 1):
            print(f"{i}. {symbol}")
        
        print(f"\n... and {len(symbols) - 10} more")
    else:
        print("Failed to fetch S&P 500 symbols")
        
        # Try to load from file
        saved_symbols = load_sp500_symbols()
        if saved_symbols:
            print(f"Loaded {len(saved_symbols)} symbols from saved file")
        else:
            print("No saved symbols file found")


if __name__ == "__main__":
    main()
