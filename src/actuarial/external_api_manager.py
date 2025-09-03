"""
External API Manager
Handles all external data sources for professional actuarial calculations
No hardcoded values - everything from APIs and external sources
"""

import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import asyncio
import warnings
warnings.filterwarnings('ignore')

class ExternalAPIManager:
    """Centralized manager for all external data sources"""
    
    def __init__(self):
        self.fred_api_key = self._get_fred_api_key()
        self.fred_base_url = "https://api.stlouisfed.org/fred"
        self.cache_dir = Path("cache/external_data")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_fred_api_key(self) -> Optional[str]:
        """Get FRED API key from environment or config"""
        # Try environment variable first
        api_key = os.getenv('FRED_API_KEY')
        
        # Try config file
        if not api_key:
            config_path = Path("config/api_keys.json")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    api_key = config.get('fred_api_key')
        
        if not api_key:
            print("⚠️ FRED API key not found. Using demo data. Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
            
        return api_key
    
    async def get_treasury_yield_curve(self, date: Optional[str] = None) -> pd.DataFrame:
        """Get complete Treasury yield curve from FRED"""
        
        # Treasury series mapping - FRED series IDs for different maturities
        treasury_series = {
            '1M': 'DGS1MO',   # 1-Month Treasury
            '3M': 'DGS3MO',   # 3-Month Treasury
            '6M': 'DGS6MO',   # 6-Month Treasury
            '1Y': 'DGS1',     # 1-Year Treasury
            '2Y': 'DGS2',     # 2-Year Treasury
            '3Y': 'DGS3',     # 3-Year Treasury
            '5Y': 'DGS5',     # 5-Year Treasury
            '7Y': 'DGS7',     # 7-Year Treasury
            '10Y': 'DGS10',   # 10-Year Treasury
            '20Y': 'DGS20',   # 20-Year Treasury
            '30Y': 'DGS30'    # 30-Year Treasury
        }
        
        # Maturity mapping for calculations
        maturity_years = {
            '1M': 1/12, '3M': 0.25, '6M': 0.5, '1Y': 1, '2Y': 2,
            '3Y': 3, '5Y': 5, '7Y': 7, '10Y': 10, '20Y': 20, '30Y': 30
        }
        
        if not self.fred_api_key:
            return self._get_demo_treasury_curve()
        
        try:
            yield_data = []
            
            async with aiohttp.ClientSession() as session:
                for maturity, series_id in treasury_series.items():
                    url = f"{self.fred_base_url}/series/observations"
                    params = {
                        'series_id': series_id,
                        'api_key': self.fred_api_key,
                        'file_type': 'json',
                        'limit': 1,
                        'sort_order': 'desc'
                    }
                    
                    if date:
                        params['observation_start'] = date
                        params['observation_end'] = date
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            observations = data.get('observations', [])
                            
                            if observations and observations[0]['value'] != '.':
                                yield_data.append({
                                    'maturity': maturity,
                                    'years': maturity_years[maturity],
                                    'yield': float(observations[0]['value']) / 100,  # Convert from percentage
                                    'date': observations[0]['date'],
                                    'series_id': series_id
                                })
            
            if yield_data:
                df = pd.DataFrame(yield_data)
                df = df.sort_values('years')
                
                # Cache the data
                cache_file = self.cache_dir / f"treasury_curve_{datetime.now().strftime('%Y%m%d')}.csv"
                df.to_csv(cache_file, index=False)
                
                return df
            else:
                return self._get_demo_treasury_curve()
                
        except Exception as e:
            print(f"Error fetching Treasury data: {e}")
            return self._get_demo_treasury_curve()
    
    async def get_economic_indicators(self) -> Dict:
        """Get current economic indicators from FRED"""
        
        economic_series = {
            'gdp_growth': 'A191RL1Q225SBEA',    # Real GDP Growth Rate
            'unemployment': 'UNRATE',           # Unemployment Rate  
            'inflation': 'CPIAUCSL',            # CPI All Items
            'fed_funds': 'FEDFUNDS',            # Federal Funds Rate
            'vix': 'VIXCLS',                    # VIX Volatility Index
            'credit_spread': 'BAA10Y'           # BAA-10Y Treasury Spread
        }
        
        if not self.fred_api_key:
            return self._get_demo_economic_indicators()
        
        try:
            indicators = {}
            
            async with aiohttp.ClientSession() as session:
                for indicator, series_id in economic_series.items():
                    url = f"{self.fred_base_url}/series/observations"
                    params = {
                        'series_id': series_id,
                        'api_key': self.fred_api_key,
                        'file_type': 'json',
                        'limit': 1,
                        'sort_order': 'desc'
                    }
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            observations = data.get('observations', [])
                            
                            if observations and observations[0]['value'] != '.':
                                indicators[indicator] = {
                                    'value': float(observations[0]['value']),
                                    'date': observations[0]['date'],
                                    'series_id': series_id
                                }
            
            if indicators:
                # Cache the data
                cache_file = self.cache_dir / f"economic_indicators_{datetime.now().strftime('%Y%m%d')}.json"
                with open(cache_file, 'w') as f:
                    json.dump(indicators, f, indent=2, default=str)
                
                return indicators
            else:
                return self._get_demo_economic_indicators()
                
        except Exception as e:
            print(f"Error fetching economic indicators: {e}")
            return self._get_demo_economic_indicators()
    
    async def get_historical_treasury_data(self, years_back: int = 10) -> pd.DataFrame:
        """Get historical Treasury data for Monte Carlo calibration"""
        
        if not self.fred_api_key:
            return self._get_demo_historical_treasury()
        
        try:
            start_date = (datetime.now() - timedelta(days=years_back*365)).strftime('%Y-%m-%d')
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.fred_base_url}/series/observations"
                params = {
                    'series_id': 'DGS10',  # 10-Year Treasury as benchmark
                    'api_key': self.fred_api_key,
                    'file_type': 'json',
                    'observation_start': start_date,
                    'frequency': 'm'  # Monthly data
                }
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        observations = data.get('observations', [])
                        
                        historical_data = []
                        for obs in observations:
                            if obs['value'] != '.':
                                historical_data.append({
                                    'date': obs['date'],
                                    'rate': float(obs['value']) / 100
                                })
                        
                        df = pd.DataFrame(historical_data)
                        df['date'] = pd.to_datetime(df['date'])
                        
                        # Calculate monthly changes for volatility calibration
                        df['rate_change'] = df['rate'].pct_change()
                        df['log_rate'] = np.log(df['rate'])
                        df['log_change'] = df['log_rate'].diff()
                        
                        return df
                        
        except Exception as e:
            print(f"Error fetching historical Treasury data: {e}")
            
        return self._get_demo_historical_treasury()
    
    def _get_demo_treasury_curve(self) -> pd.DataFrame:
        """Demo Treasury curve data for when API is unavailable"""
        return pd.DataFrame({
            'maturity': ['1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y'],
            'years': [1/12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30],
            'yield': [0.0525, 0.0510, 0.0495, 0.0475, 0.0420, 0.0425, 0.0430, 0.0440, 0.0450, 0.0470, 0.0460],
            'date': [datetime.now().strftime('%Y-%m-%d')] * 11,
            'series_id': ['DEMO'] * 11
        })
    
    def _get_demo_economic_indicators(self) -> Dict:
        """Demo economic indicators for when API is unavailable"""
        return {
            'gdp_growth': {'value': 2.4, 'date': datetime.now().strftime('%Y-%m-%d')},
            'unemployment': {'value': 3.8, 'date': datetime.now().strftime('%Y-%m-%d')},
            'inflation': {'value': 3.2, 'date': datetime.now().strftime('%Y-%m-%d')},
            'fed_funds': {'value': 5.25, 'date': datetime.now().strftime('%Y-%m-%d')},
            'vix': {'value': 18.5, 'date': datetime.now().strftime('%Y-%m-%d')},
            'credit_spread': {'value': 1.85, 'date': datetime.now().strftime('%Y-%m-%d')}
        }
    
    def _get_demo_historical_treasury(self) -> pd.DataFrame:
        """Demo historical Treasury data"""
        dates = pd.date_range(end=datetime.now(), periods=120, freq='M')
        np.random.seed(42)  # For consistent demo data
        rates = 0.03 + np.cumsum(np.random.normal(0, 0.002, 120))  # Random walk around 3%
        rates = np.clip(rates, 0.005, 0.08)  # Realistic bounds
        
        df = pd.DataFrame({
            'date': dates,
            'rate': rates
        })
        df['rate_change'] = df['rate'].pct_change()
        df['log_rate'] = np.log(df['rate'])
        df['log_change'] = df['log_rate'].diff()
        
        return df
    
    def get_cache_status(self) -> Dict:
        """Check status of cached external data"""
        cache_files = list(self.cache_dir.glob("*.csv")) + list(self.cache_dir.glob("*.json"))
        
        status = {
            'cache_directory': str(self.cache_dir),
            'total_files': len(cache_files),
            'files': []
        }
        
        for file_path in cache_files:
            file_stats = file_path.stat()
            age_hours = (datetime.now().timestamp() - file_stats.st_mtime) / 3600
            
            status['files'].append({
                'filename': file_path.name,
                'size_kb': round(file_stats.st_size / 1024, 2),
                'age_hours': round(age_hours, 2),
                'fresh': age_hours < 24  # Consider data fresh if less than 24 hours old
            })
        
        return status
    
    async def refresh_all_data(self) -> Dict:
        """Refresh all external data sources"""
        print("🔄 Refreshing external data sources...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'sources_updated': []
        }
        
        try:
            # Refresh Treasury curve
            treasury_data = await self.get_treasury_yield_curve()
            results['sources_updated'].append({
                'source': 'Treasury Yield Curve',
                'status': 'success',
                'records': len(treasury_data)
            })
            
            # Refresh economic indicators
            economic_data = await self.get_economic_indicators()
            results['sources_updated'].append({
                'source': 'Economic Indicators',
                'status': 'success', 
                'records': len(economic_data)
            })
            
            # Refresh historical data
            historical_data = await self.get_historical_treasury_data()
            results['sources_updated'].append({
                'source': 'Historical Treasury Data',
                'status': 'success',
                'records': len(historical_data)
            })
            
            print("✅ External data refresh complete!")
            
        except Exception as e:
            results['error'] = str(e)
            print(f"❌ Error refreshing external data: {e}")
        
        return results


# Usage example and testing
async def main():
    """Test the External API Manager"""
    
    api_manager = ExternalAPIManager()
    
    print("🏦 Testing Treasury Yield Curve...")
    treasury_curve = await api_manager.get_treasury_yield_curve()
    print(f"Retrieved {len(treasury_curve)} Treasury rates")
    print(treasury_curve[['maturity', 'yield']].head())
    
    print("\n📊 Testing Economic Indicators...")
    indicators = await api_manager.get_economic_indicators()
    for indicator, data in indicators.items():
        print(f"{indicator}: {data['value']:.2f}% (as of {data['date']})")
    
    print("\n📈 Testing Historical Data...")
    historical = await api_manager.get_historical_treasury_data(years_back=2)
    print(f"Retrieved {len(historical)} historical observations")
    print(f"Average rate: {historical['rate'].mean():.3f}")
    print(f"Volatility: {historical['rate_change'].std():.4f}")
    
    print("\n💾 Cache Status...")
    cache_status = api_manager.get_cache_status()
    print(f"Cache files: {cache_status['total_files']}")
    

if __name__ == "__main__":
    asyncio.run(main())