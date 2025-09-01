import pandas as pd
import numpy as np

# Create messy insurance data with various quality issues
data = {
    'Policy Number': ['POL001', 'POL002', 'POL003', ' POL004 ', 'POL005', '', 'POL006', 'POL007', 'POL008', 'POL009', 'POL010', 'Header Row Accidentally Here', 'POL011', 'POL012', 'POL013', 'POL014'],
    'Policy_Date': [' 2023-01-15 ', '2023-02-20', ' INVALID DATE', '2023-03-22', '2023/04/18', '2023-05-12', '2023-06-30', '2023-07-15', 'July 20, 2023', '2023/08/25', '2023-09-18', '', '2023-10-22', 'Invalid Format', '2023-11-30', '2023/12/08'],
    'Premium_Amount': [25000.50, 18500.75, np.nan, '$35,000.25', '22500', 19500.00, '25,500.75', 'invalid', 30500.25, '$28,750.50', 27500, '', 26500.75, 23500, '33,250.25', 21500.50],
    'Age': [35, 42, 29, 38, 45, np.nan, 55, 28, 33, 41, 39, '', 31, 46, 52, 37],
    'Gender': [' Male ', ' female', 'M', 'Female', 'MALE', 'Male', 'F', 'female', 'M', 'FEMALE', ' male ', '', 'Male', 'F', 'female', 'M'],
    'State': ['CA', 'NY', 'TX', 'FL', 'IL', 'CA', 'WA', 'OR', 'AZ', 'NV', 'CO', '', 'UT', 'ID', 'MT', 'WY'],
    'Coverage_Type': ['LIFE', 'LIFE', 'life', 'LIFE', 'Life', 'LIFE', 'HEALTH', 'LIFE', 'life insurance', 'LIFE', 'LIFE', '', 'LIFE', 'LIFE', 'Life Coverage', 'LIFE'],
    'Sum_Assured': [500000, 750000, 250000, 1000000, 500000, np.nan, 350000, 600000, 750000, 500000, 800000, '', 450000, 650000, 900000, 550000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save as Excel
df.to_excel('/Users/shijuprakash/Mr.Clean/test_messy_insurance_data.xlsx', index=False)
print("Created messy insurance test data with the following issues:")
print("- Mixed date formats")
print("- Inconsistent text case")
print("- Leading/trailing whitespace")  
print("- Invalid values (NULL, 'invalid')")
print("- Currency formatting ($, commas)")
print("- Empty rows and cells")
print("- Potential header row mixed in data")
print("- Inconsistent data types")