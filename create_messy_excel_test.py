#!/usr/bin/env python3
"""
Create a messy Excel test file to verify upload functionality
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
import random

def create_messy_excel_data():
    """Create a messy Excel file similar to real insurance data"""
    
    # Create messy insurance data
    data = []
    
    # Add some header rows (common issue)
    data.append(["", "", "", "INSURANCE CLAIMS REPORT", "", "", ""])
    data.append(["", "", "", "Generated: " + str(date.today()), "", "", ""])
    data.append(["", "", "", "", "", "", ""])  # Empty row
    
    # Real header
    data.append(["Policy Number", "Claim Date", "Claim Amount", "Status", "Gender", "Age", "Product Type"])
    
    # Add messy data rows
    sample_data = [
        ["POL001", "2023-01-15", 15000.50, "OPEN", "M", 45, "Term Life"],
        ["POL002", "01/22/2023", "25,000", "closed", "f", 38, "WHOLE LIFE"],
        ["POL003", "2023-3-8", "$35,000.75", "Open", "Male", 52, "universal life"],
        ["", "", "", "", "", "", ""],  # Empty row
        ["POL004", "March 15, 2023", "12000", "CLOSED", "FEMALE", 29, "Term Life"],
        ["POL005", "2023/04/20", "45000.00", "open", "m", 61, "Variable Life"],
        ["POL006", "4/30/23", "22,500", "Closed", "F", "45", "term life"],
        ["POL007", "2023-05-10", "invalid", "pending", "0", 33, "Whole Life"],
        ["TOTAL:", "", "154,000.25", "", "", "", ""],  # Summary row
    ]
    
    data.extend(sample_data)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to Excel
    filename = "/Users/shijuprakash/Mr.Clean/test_messy_data.xlsx"
    df.to_excel(filename, index=False, header=False)
    
    print(f"✅ Created messy Excel test file: {filename}")
    print(f"📊 Data includes:")
    print(f"  • Header rows at top (common Excel issue)")
    print(f"  • Mixed date formats: 2023-01-15, 01/22/2023, March 15, 2023")
    print(f"  • Mixed case: OPEN/closed/Open, M/f/Male/FEMALE")
    print(f"  • Currency formatting: $35,000.75, 25,000") 
    print(f"  • Empty rows and invalid data")
    print(f"  • Summary rows mixed with data")
    
    return filename

if __name__ == "__main__":
    filename = create_messy_excel_data()
    print(f"\n🎯 Upload this file to test: {filename}")