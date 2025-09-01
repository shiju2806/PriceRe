"""
Comprehensive Test Data Generator
Creates realistic messy datasets for testing cleaning algorithms
"""

import pandas as pd
import polars as pl
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import random
import string
from pathlib import Path
import tempfile

class MessType(Enum):
    """Types of data messiness"""
    EMPTY_ROWS = "empty_rows"
    HEADER_MISPLACED = "header_misplaced" 
    FOOTER_JUNK = "footer_junk"
    MIXED_DATA_TYPES = "mixed_data_types"
    MERGED_CELLS = "merged_cells"
    MULTI_TABLE = "multi_table"
    INCONSISTENT_FORMATS = "inconsistent_formats"
    DUPLICATE_HEADERS = "duplicate_headers"
    SUMMARY_ROWS = "summary_rows"
    PAGINATION_ARTIFACTS = "pagination_artifacts"

@dataclass
class TestCase:
    """Test case with messy data and ground truth"""
    name: str
    messy_df: pl.DataFrame
    clean_df: pl.DataFrame  # Ground truth
    junk_row_indices: List[int]  # Which rows should be removed
    mess_types: List[MessType]
    description: str
    expected_accuracy: float = 0.9  # Minimum expected cleaning accuracy

class TestDataGenerator:
    """Generate comprehensive test datasets for cleaning validation"""
    
    def __init__(self, random_seed: int = 42):
        random.seed(random_seed)
        np.random.seed(random_seed)
        
    def create_comprehensive_test_suite(self) -> Dict[str, TestCase]:
        """Create full test suite covering all scenarios"""
        
        test_suite = {}
        
        # Basic junk removal tests
        test_suite['empty_rows'] = self._create_empty_rows_test()
        test_suite['header_misplaced'] = self._create_misplaced_headers_test()
        test_suite['footer_junk'] = self._create_footer_junk_test()
        
        # Complex structural tests
        test_suite['multi_table'] = self._create_multi_table_test()
        test_suite['mixed_formats'] = self._create_mixed_format_test()
        test_suite['summary_rows'] = self._create_summary_rows_test()
        
        # Edge cases
        test_suite['pagination_artifacts'] = self._create_pagination_test()
        test_suite['merged_cells'] = self._create_merged_cells_test()
        test_suite['duplicate_headers'] = self._create_duplicate_headers_test()
        
        # Domain-specific tests
        test_suite['insurance_realistic'] = self._create_realistic_insurance_test()
        test_suite['financial_messy'] = self._create_financial_mess_test()
        
        # Scale tests
        test_suite['large_file'] = self._create_large_file_test()
        
        # Multi-language tests
        test_suite['multilingual'] = self._create_multilingual_test()
        
        return test_suite
    
    def _create_base_clean_data(self, domain: str = "insurance", rows: int = 20) -> pl.DataFrame:
        """Create clean base data for different domains"""
        
        if domain == "insurance":
            return self._create_clean_insurance_data(rows)
        elif domain == "financial":
            return self._create_clean_financial_data(rows)
        else:
            return self._create_generic_clean_data(rows)
    
    def _create_clean_insurance_data(self, rows: int) -> pl.DataFrame:
        """Create clean insurance policy data"""
        
        # Generate dates as strings to avoid schema conflicts
        dates = pd.date_range('2023-01-01', periods=rows, freq='D')
        date_strings = [d.strftime('%Y-%m-%d') for d in dates]
        
        data = {
            'Policy_Number': [f'POL{i:05d}' for i in range(1, rows+1)],
            'Policy_Date': date_strings,
            'Premium_Amount': np.random.uniform(15000, 50000, rows).round(2),
            'Age': np.random.randint(25, 65, rows),
            'Gender': np.random.choice(['Male', 'Female'], rows),
            'State': np.random.choice(['CA', 'NY', 'TX', 'FL', 'IL'], rows),
            'Coverage_Type': np.random.choice(['LIFE', 'HEALTH', 'AUTO'], rows),
            'Sum_Assured': np.random.uniform(250000, 1000000, rows).round(0)
        }
        
        return pl.DataFrame(data)
    
    def _create_clean_financial_data(self, rows: int) -> pl.DataFrame:
        """Create clean financial transaction data"""
        
        # Generate dates as strings 
        dates = pd.date_range('2023-01-01', periods=rows, freq='H')
        date_strings = [d.strftime('%Y-%m-%d %H:%M:%S') for d in dates]
        
        data = {
            'Transaction_ID': [f'TXN{i:06d}' for i in range(1, rows+1)],
            'Date': date_strings,
            'Amount': np.random.uniform(-5000, 10000, rows).round(2),
            'Account': [f'ACC{random.randint(1000,9999)}' for _ in range(rows)],
            'Category': np.random.choice(['Food', 'Transport', 'Entertainment', 'Bills'], rows),
            'Description': [f'Transaction {i}' for i in range(rows)]
        }
        
        return pl.DataFrame(data)
    
    def _create_generic_clean_data(self, rows: int) -> pl.DataFrame:
        """Create generic clean data"""
        
        # Generate dates as strings
        dates = pd.date_range('2023-01-01', periods=rows)
        date_strings = [d.strftime('%Y-%m-%d') for d in dates]
        
        data = {
            'ID': range(1, rows+1),
            'Name': [f'Item_{i}' for i in range(1, rows+1)],
            'Value': np.random.uniform(100, 1000, rows).round(2),
            'Category': np.random.choice(['A', 'B', 'C'], rows),
            'Date': date_strings
        }
        
        return pl.DataFrame(data)
    
    def _create_empty_rows_test(self) -> TestCase:
        """Test case: Empty rows scattered throughout data"""
        
        clean_df = self._create_base_clean_data("insurance", 15)
        
        # Create messy version with empty rows
        messy_data = []
        junk_indices = []
        
        for i, row in enumerate(clean_df.to_pandas().iterrows()):
            messy_data.append(row[1].values)
            
            # Add empty rows randomly
            if random.random() < 0.2:  # 20% chance
                messy_data.append([None] * len(clean_df.columns))
                junk_indices.append(len(messy_data) - 1)
        
        # Add some empty rows at the end
        for _ in range(3):
            messy_data.append([None] * len(clean_df.columns))
            junk_indices.append(len(messy_data) - 1)
        
        messy_df = pl.DataFrame(messy_data, schema=clean_df.columns)
        
        return TestCase(
            name="empty_rows",
            messy_df=messy_df,
            clean_df=clean_df,
            junk_row_indices=junk_indices,
            mess_types=[MessType.EMPTY_ROWS],
            description="Scattered empty rows that should be removed",
            expected_accuracy=0.95
        )
    
    def _create_misplaced_headers_test(self) -> TestCase:
        """Test case: Headers in wrong position"""
        
        clean_df = self._create_base_clean_data("insurance", 10)
        
        # Create messy version with headers in wrong place
        messy_rows = []
        
        # Add junk title rows at top
        messy_rows.append(["Insurance Report Q3 2023", None, None, None, None, None, None, None])
        messy_rows.append([None, None, None, None, None, None, None, None])  # Empty row
        
        # Add actual headers (row index 2)
        messy_rows.append(clean_df.columns)
        
        # Add data
        for row in clean_df.to_pandas().iterrows():
            messy_rows.append(row[1].values)
        
        messy_df = pl.DataFrame(messy_rows, schema=clean_df.columns)
        
        return TestCase(
            name="header_misplaced",
            messy_df=messy_df,
            clean_df=clean_df,
            junk_row_indices=[0, 1, 2],  # Title, empty row, header row
            mess_types=[MessType.HEADER_MISPLACED],
            description="Headers in wrong position with title rows",
            expected_accuracy=0.9
        )
    
    def _create_footer_junk_test(self) -> TestCase:
        """Test case: Footer junk at bottom"""
        
        clean_df = self._create_base_clean_data("insurance", 12)
        
        # Start with clean data
        messy_rows = [row for row in clean_df.to_pandas().values]
        
        # Add footer junk
        footer_junk = [
            ["Total Policies: 12", None, None, None, None, None, None, None],
            ["Generated on 2023-10-15", None, None, None, None, None, None, None],
            ["Page 1 of 1", None, None, None, None, None, None, None],
            ["Confidential", None, None, None, None, None, None, None]
        ]
        
        junk_indices = []
        for junk_row in footer_junk:
            messy_rows.append(junk_row)
            junk_indices.append(len(messy_rows) - 1)
        
        messy_df = pl.DataFrame(messy_rows, schema=clean_df.columns)
        
        return TestCase(
            name="footer_junk",
            messy_df=messy_df,
            clean_df=clean_df,
            junk_row_indices=junk_indices,
            mess_types=[MessType.FOOTER_JUNK],
            description="Footer metadata rows at bottom",
            expected_accuracy=0.9
        )
    
    def _create_summary_rows_test(self) -> TestCase:
        """Test case: Summary/total rows mixed in data"""
        
        clean_df = self._create_base_clean_data("insurance", 20)
        
        messy_rows = []
        junk_indices = []
        clean_row_count = 0
        
        for i, row in enumerate(clean_df.to_pandas().iterrows()):
            messy_rows.append(row[1].values)
            clean_row_count += 1
            
            # Add summary row every 5 data rows
            if (i + 1) % 5 == 0:
                summary_row = [f"Subtotal (5 policies)", None, f"${random.randint(100000, 200000)}", None, None, None, None, None]
                messy_rows.append(summary_row)
                junk_indices.append(len(messy_rows) - 1)
        
        # Add final total
        total_row = ["TOTAL", None, f"${random.randint(500000, 800000)}", None, None, None, None, None]
        messy_rows.append(total_row)
        junk_indices.append(len(messy_rows) - 1)
        
        messy_df = pl.DataFrame(messy_rows, schema=clean_df.columns)
        
        return TestCase(
            name="summary_rows",
            messy_df=messy_df,
            clean_df=clean_df,
            junk_row_indices=junk_indices,
            mess_types=[MessType.SUMMARY_ROWS],
            description="Summary and total rows mixed with data",
            expected_accuracy=0.85
        )
    
    def _create_multi_table_test(self) -> TestCase:
        """Test case: Multiple tables in one dataset"""
        
        # First table: Insurance policies
        clean_df1 = self._create_base_clean_data("insurance", 8)
        
        # Second table: Claims data (different structure)
        claims_data = {
            'Claim_ID': [f'CLM{i:04d}' for i in range(1, 6)],
            'Claim_Date': pd.date_range('2023-06-01', periods=5),
            'Amount_Claimed': np.random.uniform(1000, 15000, 5).round(2),
            'Status': np.random.choice(['Approved', 'Pending'], 5)
        }
        clean_df2 = pl.DataFrame(claims_data)
        
        # We'll only clean the first table, so clean_df = clean_df1
        clean_df = clean_df1
        
        # Create messy version with both tables
        messy_rows = []
        junk_indices = []
        
        # Add first table with headers
        messy_rows.append(clean_df1.columns)
        junk_indices.append(0)  # Header row is junk
        
        for row in clean_df1.to_pandas().values:
            messy_rows.append(row)
        
        # Add separator
        messy_rows.append(["------- CLAIMS DATA -------", None, None, None, None, None, None, None])
        junk_indices.append(len(messy_rows) - 1)
        
        # Add second table (all junk for our purposes)
        messy_rows.append(["Claim_ID", "Claim_Date", "Amount_Claimed", "Status", None, None, None, None])
        junk_indices.append(len(messy_rows) - 1)
        
        for row in clean_df2.to_pandas().values:
            padded_row = list(row) + [None] * (8 - 4)  # Pad to match columns
            messy_rows.append(padded_row)
            junk_indices.append(len(messy_rows) - 1)
        
        messy_df = pl.DataFrame(messy_rows, schema=clean_df.columns)
        
        return TestCase(
            name="multi_table",
            messy_df=messy_df,
            clean_df=clean_df,
            junk_row_indices=junk_indices,
            mess_types=[MessType.MULTI_TABLE],
            description="Multiple tables with different structures",
            expected_accuracy=0.8
        )
    
    def _create_realistic_insurance_test(self) -> TestCase:
        """Test case: Realistic insurance Excel mess"""
        
        clean_df = self._create_base_clean_data("insurance", 25)
        
        messy_rows = []
        junk_indices = []
        
        # Realistic Excel mess
        mess_patterns = [
            ["ACME Insurance Company", None, None, None, None, None, None, None],
            ["Policy Report - Q3 2023", None, None, None, None, None, None, None],
            ["Generated: October 15, 2023", None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None],  # Empty
            ["Policy Details:", None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None],  # Empty
        ]
        
        # Add initial junk
        for i, row in enumerate(mess_patterns):
            messy_rows.append(row)
            junk_indices.append(i)
        
        # Add headers
        messy_rows.append(clean_df.columns)
        junk_indices.append(len(messy_rows) - 1)  # Headers are junk too
        
        # Add data with occasional issues
        for i, row in enumerate(clean_df.to_pandas().values):
            messy_rows.append(row)
            
            # Add occasional separator
            if (i + 1) % 10 == 0:
                messy_rows.append(["----------", "----------", "----------", "----------", "----------", "----------", "----------", "----------"])
                junk_indices.append(len(messy_rows) - 1)
        
        # Add footer
        footer_mess = [
            [None, None, None, None, None, None, None, None],  # Empty
            [f"Total Policies: {len(clean_df)}", None, None, None, None, None, None, None],
            ["End of Report", None, None, None, None, None, None, None],
            ["*** CONFIDENTIAL ***", None, None, None, None, None, None, None]
        ]
        
        for row in footer_mess:
            messy_rows.append(row)
            junk_indices.append(len(messy_rows) - 1)
        
        messy_df = pl.DataFrame(messy_rows, schema=clean_df.columns)
        
        return TestCase(
            name="insurance_realistic", 
            messy_df=messy_df,
            clean_df=clean_df,
            junk_row_indices=junk_indices,
            mess_types=[MessType.HEADER_MISPLACED, MessType.FOOTER_JUNK, MessType.SUMMARY_ROWS],
            description="Realistic insurance Excel export with typical mess",
            expected_accuracy=0.85
        )
    
    def _create_large_file_test(self) -> TestCase:
        """Test case: Large file performance test"""
        
        clean_df = self._create_base_clean_data("insurance", 1000)  # 1K rows
        
        # Add scattered junk (10% junk ratio)
        messy_rows = []
        junk_indices = []
        
        for i, row in enumerate(clean_df.to_pandas().values):
            messy_rows.append(row)
            
            # Add junk every 10 rows
            if i % 10 == 0:
                junk_row = [f"--- Section {i//10} ---", None, None, None, None, None, None, None]
                messy_rows.append(junk_row)
                junk_indices.append(len(messy_rows) - 1)
        
        messy_df = pl.DataFrame(messy_rows, schema=clean_df.columns)
        
        return TestCase(
            name="large_file",
            messy_df=messy_df,
            clean_df=clean_df,
            junk_row_indices=junk_indices,
            mess_types=[MessType.SUMMARY_ROWS],
            description="Large file with scattered junk for performance testing",
            expected_accuracy=0.9
        )
    
    def _create_multilingual_test(self) -> TestCase:
        """Test case: Multi-language data"""
        
        # Create base data with mixed languages
        data = {
            'Policy_Number': [f'POL{i:03d}' for i in range(1, 11)],
            'Fecha_Poliza': ['2023-01-15', '2023-02-20', '2023-03-10', 'Invalid Date', '2023-04-05', '2023-05-12', '2023-06-08', '2023-07-22', '2023-08-15', '2023-09-30'],
            'Prima_Mensual': [25000.50, 18500.75, np.nan, 35000.25, 22500, 19500.00, 25500.75, 30500.25, 28750.50, 27500],
            'Edad': [35, 42, 29, 38, 45, np.nan, 55, 28, 33, 41],
            'Género': ['Masculino', 'Femenino', 'M', 'Mujer', 'Hombre', 'F', 'Femenino', 'M', 'Masculino', 'F'],
            'Estado': ['CA', 'NY', 'TX', 'FL', 'IL', 'WA', 'OR', 'AZ', 'NV', 'CO']
        }
        
        clean_df = pl.DataFrame(data)
        
        # Add Spanish junk
        messy_rows = []
        junk_indices = []
        
        # Spanish headers and junk
        spanish_junk = [
            ["Reporte de Pólizas - Trimestre 3", None, None, None, None, None],
            ["Compañía: ACME Seguros", None, None, None, None, None],
            [None, None, None, None, None, None],  # Empty
        ]
        
        for i, row in enumerate(spanish_junk):
            messy_rows.append(row)
            junk_indices.append(i)
        
        # Add data
        for row in clean_df.to_pandas().values:
            messy_rows.append(row)
        
        # Spanish footer
        messy_rows.append(["Total de Pólizas: 10", None, None, None, None, None])
        junk_indices.append(len(messy_rows) - 1)
        
        messy_df = pl.DataFrame(messy_rows, schema=clean_df.columns)
        
        return TestCase(
            name="multilingual",
            messy_df=messy_df,
            clean_df=clean_df,
            junk_row_indices=junk_indices,
            mess_types=[MessType.HEADER_MISPLACED, MessType.FOOTER_JUNK],
            description="Multi-language data with Spanish junk rows",
            expected_accuracy=0.8
        )
    
    def _create_mixed_format_test(self) -> TestCase:
        """Test case: Mixed data formats within columns"""
        
        # Create data with mixed formats
        data = {
            'ID': ['001', 'ID-002', 'NUM003', '4', 'ITEM-005'],
            'Date': ['2023-01-01', '01/02/2023', 'March 3, 2023', '2023-4-4', 'invalid'],
            'Amount': ['$1,000.50', '2000.75', '3,500', 'USD 4000', 'invalid'],
            'Status': ['Active', 'ACTIVE', 'active', 'A', 'Valid']
        }
        
        clean_df = pl.DataFrame(data)
        
        # This test focuses on format inconsistency, not junk rows
        # So messy_df is the same, but we expect normalization
        messy_df = clean_df.clone()
        
        return TestCase(
            name="mixed_formats",
            messy_df=messy_df,
            clean_df=clean_df,
            junk_row_indices=[],  # No rows to remove, just format fixes
            mess_types=[MessType.INCONSISTENT_FORMATS],
            description="Mixed data formats requiring normalization",
            expected_accuracy=0.95
        )
    
    def _create_pagination_test(self) -> TestCase:
        """Test case: Pagination artifacts"""
        
        clean_df = self._create_base_clean_data("insurance", 15)
        
        messy_rows = []
        junk_indices = []
        
        # Add first batch of data
        for i, row in enumerate(clean_df.to_pandas().values[:8]):
            messy_rows.append(row)
        
        # Add page break artifacts
        page_artifacts = [
            ["--- Page 1 End ---", None, None, None, None, None, None, None],
            [None, None, None, None, None, None, None, None],  # Empty
            ["--- Page 2 Start ---", None, None, None, None, None, None, None],
            ["Continued from previous page...", None, None, None, None, None, None, None]
        ]
        
        for artifact in page_artifacts:
            messy_rows.append(artifact)
            junk_indices.append(len(messy_rows) - 1)
        
        # Add remaining data
        for row in clean_df.to_pandas().values[8:]:
            messy_rows.append(row)
        
        messy_df = pl.DataFrame(messy_rows, schema=clean_df.columns)
        
        return TestCase(
            name="pagination_artifacts",
            messy_df=messy_df,
            clean_df=clean_df,
            junk_row_indices=junk_indices,
            mess_types=[MessType.PAGINATION_ARTIFACTS],
            description="Page break and continuation artifacts",
            expected_accuracy=0.9
        )
    
    def _create_merged_cells_test(self) -> TestCase:
        """Test case: Simulated merged cell effects"""
        
        clean_df = self._create_base_clean_data("insurance", 10)
        
        # Simulate merged cells by repeating values or using None
        messy_rows = []
        junk_indices = []
        
        # Add data with merged cell simulation
        current_group = None
        for i, row in enumerate(clean_df.to_pandas().values):
            if i % 3 == 0:  # Start new group every 3 rows
                current_group = f"Group {i//3 + 1}"
                # Add group header row (junk)
                group_row = [current_group, None, None, None, None, None, None, None]
                messy_rows.append(group_row)
                junk_indices.append(len(messy_rows) - 1)
            
            messy_rows.append(row)
        
        messy_df = pl.DataFrame(messy_rows, schema=clean_df.columns)
        
        return TestCase(
            name="merged_cells",
            messy_df=messy_df,
            clean_df=clean_df,
            junk_row_indices=junk_indices,
            mess_types=[MessType.MERGED_CELLS],
            description="Simulated merged cell grouping effects",
            expected_accuracy=0.85
        )
    
    def _create_duplicate_headers_test(self) -> TestCase:
        """Test case: Duplicate headers scattered in data"""
        
        clean_df = self._create_base_clean_data("insurance", 12)
        
        messy_rows = []
        junk_indices = []
        
        # Add initial header
        messy_rows.append(clean_df.columns)
        junk_indices.append(0)
        
        # Add data with duplicate headers every 4 rows
        for i, row in enumerate(clean_df.to_pandas().values):
            messy_rows.append(row)
            
            if (i + 1) % 4 == 0 and i < len(clean_df) - 1:  # Don't add at the very end
                # Add duplicate header
                messy_rows.append(clean_df.columns)
                junk_indices.append(len(messy_rows) - 1)
        
        messy_df = pl.DataFrame(messy_rows, schema=clean_df.columns)
        
        return TestCase(
            name="duplicate_headers",
            messy_df=messy_df,
            clean_df=clean_df,
            junk_row_indices=junk_indices,
            mess_types=[MessType.DUPLICATE_HEADERS],
            description="Duplicate header rows scattered throughout data",
            expected_accuracy=0.9
        )
    
    def save_test_suite(self, test_suite: Dict[str, TestCase], output_dir: Path) -> None:
        """Save test suite to files for inspection and debugging"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        for test_name, test_case in test_suite.items():
            test_dir = output_dir / test_name
            test_dir.mkdir(exist_ok=True)
            
            # Save messy data
            test_case.messy_df.write_csv(test_dir / "messy.csv")
            
            # Save clean ground truth
            test_case.clean_df.write_csv(test_dir / "clean_ground_truth.csv")
            
            # Save metadata
            metadata = {
                'description': test_case.description,
                'junk_row_indices': test_case.junk_row_indices,
                'mess_types': [mt.value for mt in test_case.mess_types],
                'expected_accuracy': test_case.expected_accuracy,
                'messy_shape': test_case.messy_df.shape,
                'clean_shape': test_case.clean_df.shape
            }
            
            with open(test_dir / "metadata.json", 'w') as f:
                import json
                json.dump(metadata, f, indent=2)

# Convenience function
def generate_test_suite(save_to_disk: bool = False, output_dir: str = "./test_data") -> Dict[str, TestCase]:
    """Generate comprehensive test suite"""
    
    generator = TestDataGenerator()
    test_suite = generator.create_comprehensive_test_suite()
    
    if save_to_disk:
        generator.save_test_suite(test_suite, Path(output_dir))
        print(f"Test suite saved to {output_dir}")
    
    return test_suite