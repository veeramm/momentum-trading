#!/usr/bin/env python
"""
Fetch Current Dow 30 Tickers

This script fetches the current Dow Jones Industrial Average (DJIA) 30 components
and saves them for use in fundamental data fetching.
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


def fetch_dow30_symbols():
    """
    Fetch current Dow 30 symbols from Wikipedia.
    
    Returns:
        Tuple of (symbols list, companies info list)
    """
    url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    
    try:
        # Fetch the page
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the component table
        # The table usually has 'wikitable' class
        tables = soup.find_all('table', {'class': 'wikitable'})
        
        # Look for the table with company information
        dow_table = None
        for table in tables:
            # Check if this table contains stock symbols
            headers = table.find_all('th')
            header_text = [h.text.strip().lower() for h in headers]
            if any('symbol' in h for h in header_text):
                dow_table = table
                break
        
        if not dow_table:
            print("Could not find Dow 30 table on Wikipedia")
            return None, None
        
        # Parse table to DataFrame
        df = pd.read_html(str(dow_table))[0]
        
        # Find the column containing symbols
        symbol_col = None
        for col in df.columns:
            if 'symbol' in str(col).lower():
                symbol_col = col
                break
        
        if not symbol_col:
            print("Could not find symbol column")
            return None, None
        
        # Extract symbols
        symbols = df[symbol_col].tolist()
        
        # Clean symbols
        symbols = [str(symbol).strip() for symbol in symbols]
        
        # Get additional info
        companies_info = []
        for idx, row in df.iterrows():
            company_info = {
                'symbol': str(row[symbol_col]).strip(),
                'name': str(row.get('Company', '')).strip(),
                'exchange': str(row.get('Exchange', 'NYSE')).strip(),
                'industry': str(row.get('Industry', '')).strip(),
                'date_added': str(row.get('Date added', '')).strip()
            }
            companies_info.append(company_info)
        
        return symbols, companies_info
        
    except Exception as e:
        print(f"Error fetching Dow 30 symbols: {e}")
        return None, None


def save_dow30_data(symbols, companies_info):
    """
    Save Dow 30 data to files.
    
    Args:
        symbols: List of symbols
        companies_info: List of company information dictionaries
    """
    # Create data directory if it doesn't exist
    data_dir = project_root / 'data' / 'universe'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save symbols list
    symbols_file = data_dir / 'dow30_symbols.json'
    with open(symbols_file, 'w') as f:
        json.dump({
            'symbols': symbols,
            'count': len(symbols),
            'updated': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"Saved {len(symbols)} symbols to {symbols_file}")
    
    # Save detailed company info
    if companies_info:
        info_file = data_dir / 'dow30_companies.json'
        with open(info_file, 'w') as f:
            json.dump({
                'companies': companies_info,
                'count': len(companies_info),
                'updated': datetime.now().isoformat()
            }, f, indent=2)
        
        # Also save as CSV for easy viewing
        csv_file = data_dir / 'dow30_companies.csv'
        df = pd.DataFrame(companies_info)
        df.to_csv(csv_file, index=False)
        
        print(f"Saved company info to {info_file} and {csv_file}")


def load_dow30_symbols():
    """
    Load Dow 30 symbols from saved file.
    
    Returns:
        List of symbols or None if file doesn't exist
    """
    symbols_file = project_root / 'data' / 'universe' / 'dow30_symbols.json'
    
    if symbols_file.exists():
        with open(symbols_file, 'r') as f:
            data = json.load(f)
            return data['symbols']
    
    return None


def main():
    """Main function"""
    print("Fetching current Dow 30 constituents...")
    
    # Fetch symbols
    symbols, companies_info = fetch_dow30_symbols()
    
    if symbols:
        print(f"Successfully fetched {len(symbols)} Dow 30 symbols")
        
        # Save to files
        save_dow30_data(symbols, companies_info)
        
        # Display symbols
        print("\nDow 30 Symbols:")
        for i, symbol in enumerate(symbols, 1):
            print(f"{i:2d}. {symbol}")
    else:
        print("Failed to fetch Dow 30 symbols")
        
        # Try to load from file
        saved_symbols = load_dow30_symbols()
        if saved_symbols:
            print(f"Loaded {len(saved_symbols)} symbols from saved file")
            print("\nDow 30 Symbols (from cache):")
            for i, symbol in enumerate(saved_symbols, 1):
                print(f"{i:2d}. {symbol}")
        else:
            print("No saved symbols file found")


if __name__ == "__main__":
    main()
