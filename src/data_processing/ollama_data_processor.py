"""
Ollama-Powered Data Processing Engine
Uses local Llama 3.2 to understand and process any uploaded data format
Provides intelligent data interpretation and actuarial analysis
"""

import requests
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging
from datetime import datetime
import io
import chardet
import sqlite3
from dataclasses import dataclass

@dataclass
class DataProcessingResult:
    """Result of data processing operation"""
    success: bool
    data_type: str
    standardized_data: Optional[pd.DataFrame]
    quality_score: float
    issues: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]

class OllamaClient:
    """Client for local Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "llama3.2"
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate response from Ollama"""
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            response = requests.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            
            result = response.json()
            return {
                'success': True,
                'response': result.get('response', ''),
                'tokens': result.get('eval_count', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Ollama API error: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': ''
            }

class IntelligentDataProcessor:
    """AI-powered data processor using Ollama"""
    
    def __init__(self):
        self.ollama = OllamaClient()
        self.logger = logging.getLogger(__name__)
        
        # Actuarial knowledge base
        self.actuarial_context = """
        You are an expert actuarial data analyst with deep knowledge of:
        - Life insurance and reinsurance data structures
        - Mortality tables and experience analysis
        - Premium calculations and reserve methodologies
        - Claims processing and underwriting data
        - Policy administration systems
        - Economic and investment data
        - Regulatory requirements (GAAP, Statutory, IFRS 17)
        
        Always provide precise, technically accurate actuarial analysis.
        """
    
    def detect_file_encoding(self, file_content: bytes) -> str:
        """Detect file encoding"""
        detected = chardet.detect(file_content)
        return detected.get('encoding', 'utf-8')
    
    def read_file_intelligently(self, file_path: Union[str, Path], file_content: bytes = None) -> pd.DataFrame:
        """Intelligently read any file format"""
        
        if file_content:
            encoding = self.detect_file_encoding(file_content)
            content = file_content.decode(encoding)
        else:
            with open(file_path, 'rb') as f:
                file_content = f.read()
                encoding = self.detect_file_encoding(file_content)
                content = file_content.decode(encoding)
        
        file_path = Path(file_path) if file_path else Path("uploaded_file")
        
        # Try different parsing methods
        try:
            # CSV variations
            for separator in [',', ';', '|', '\t']:
                try:
                    df = pd.read_csv(io.StringIO(content), sep=separator)
                    if len(df.columns) > 1:  # Valid separation
                        self.logger.info(f"Successfully parsed as CSV with separator '{separator}'")
                        return df
                except:
                    continue
            
            # Excel files
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(io.BytesIO(file_content))
                self.logger.info("Successfully parsed as Excel file")
                return df
            
            # JSON
            if file_path.suffix.lower() == '.json':
                data = json.loads(content)
                df = pd.json_normalize(data)
                self.logger.info("Successfully parsed as JSON file")
                return df
            
            # Fixed width or other formats - let Ollama help
            return self._parse_with_ollama_help(content)
            
        except Exception as e:
            self.logger.error(f"Failed to parse file: {e}")
            raise ValueError(f"Could not parse file: {e}")
    
    def _parse_with_ollama_help(self, content: str) -> pd.DataFrame:
        """Use Ollama to help parse unusual formats"""
        
        # Show Ollama a sample of the data
        sample = content[:2000]  # First 2000 characters
        
        prompt = f"""
        Analyze this data format and help me parse it:
        
        DATA SAMPLE:
        {sample}
        
        Please identify:
        1. What type of data format this is
        2. How to parse it into structured rows and columns
        3. What the columns represent
        4. Suggested column names
        
        Respond in JSON format:
        {{
            "format_type": "csv|fixed_width|xml|other",
            "separator": "delimiter if applicable",
            "columns": ["col1", "col2", "col3"],
            "data_type": "mortality|policy|claims|financial|other",
            "parsing_instructions": "specific instructions"
        }}
        """
        
        result = self.ollama.generate(prompt, self.actuarial_context)
        
        if result['success']:
            try:
                # Extract JSON from response
                response_text = result['response']
                # Find JSON in response
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = response_text[start:end]
                    parsing_info = json.loads(json_str)
                    
                    # Try to parse based on Ollama's advice
                    if parsing_info.get('format_type') == 'csv':
                        separator = parsing_info.get('separator', ',')
                        df = pd.read_csv(io.StringIO(content), sep=separator)
                        return df
                    
            except Exception as e:
                self.logger.error(f"Could not use Ollama parsing advice: {e}")
        
        # Fallback: assume CSV with comma separator
        return pd.read_csv(io.StringIO(content))
    
    def analyze_data_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Use Ollama to analyze data structure and identify data type"""
        
        # Create data summary for Ollama
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'sample_data': df.head(3).to_dict('records'),
            'null_counts': df.isnull().sum().to_dict(),
            'unique_counts': {col: df[col].nunique() for col in df.columns}
        }
        
        prompt = f"""
        Analyze this dataset and identify what type of actuarial/insurance data it contains:
        
        DATASET SUMMARY:
        - Shape: {summary['shape'][0]} rows, {summary['shape'][1]} columns
        - Columns: {summary['columns']}
        - Sample data: {json.dumps(summary['sample_data'], indent=2, default=str)}
        
        Please analyze and respond in JSON format:
        {{
            "data_type": "mortality_table|policy_data|claims_data|premium_transactions|economic_data|underwriting_data|financial_statements|other",
            "confidence": 0.95,
            "description": "Brief description of what this data represents",
            "key_fields": ["field1", "field2"],
            "quality_issues": ["issue1", "issue2"],
            "standardization_needed": ["action1", "action2"],
            "actuarial_use_cases": ["use_case1", "use_case2"]
        }}
        """
        
        result = self.ollama.generate(prompt, self.actuarial_context)
        
        if result['success']:
            try:
                # Extract JSON from response
                response_text = result['response']
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = response_text[start:end]
                    analysis = json.loads(json_str)
                    return analysis
            except Exception as e:
                self.logger.error(f"Could not parse Ollama analysis: {e}")
        
        # Fallback analysis
        return {
            "data_type": "unknown",
            "confidence": 0.5,
            "description": "Could not automatically identify data type",
            "key_fields": list(df.columns)[:5],
            "quality_issues": [],
            "standardization_needed": [],
            "actuarial_use_cases": []
        }
    
    def standardize_data(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Standardize data format using Ollama intelligence"""
        
        # Create standardization prompt based on data type
        sample_data = df.head(5).to_string()
        
        prompt = f"""
        Standardize this {data_type} data according to actuarial best practices:
        
        CURRENT DATA:
        {sample_data}
        
        Please provide standardization instructions in JSON format:
        {{
            "column_mappings": {{
                "old_column_name": "standard_column_name"
            }},
            "data_type_conversions": {{
                "column_name": "target_type"
            }},
            "value_standardizations": {{
                "column_name": {{
                    "old_value": "new_value"
                }}
            }},
            "date_formats": {{
                "column_name": "target_format"
            }},
            "validation_rules": {{
                "column_name": "validation_rule"
            }}
        }}
        """
        
        result = self.ollama.generate(prompt, self.actuarial_context)
        
        if result['success']:
            try:
                # Extract JSON from response
                response_text = result['response']
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = response_text[start:end]
                    standardization = json.loads(json_str)
                    
                    # Apply standardizations
                    df_std = df.copy()
                    
                    # Column mappings
                    if 'column_mappings' in standardization:
                        df_std = df_std.rename(columns=standardization['column_mappings'])
                    
                    # Value standardizations
                    if 'value_standardizations' in standardization:
                        for col, mappings in standardization['value_standardizations'].items():
                            if col in df_std.columns:
                                df_std[col] = df_std[col].replace(mappings)
                    
                    # Data type conversions
                    if 'data_type_conversions' in standardization:
                        for col, dtype in standardization['data_type_conversions'].items():
                            if col in df_std.columns:
                                try:
                                    if dtype == 'datetime':
                                        df_std[col] = pd.to_datetime(df_std[col])
                                    elif dtype == 'numeric':
                                        df_std[col] = pd.to_numeric(df_std[col], errors='coerce')
                                    elif dtype == 'string':
                                        df_std[col] = df_std[col].astype(str)
                                except Exception as e:
                                    self.logger.warning(f"Could not convert {col} to {dtype}: {e}")
                    
                    return df_std
                    
            except Exception as e:
                self.logger.error(f"Could not apply standardizations: {e}")
        
        return df
    
    def assess_data_quality(self, df: pd.DataFrame, data_type: str) -> Dict[str, Any]:
        """Assess data quality using actuarial standards"""
        
        # Basic quality metrics
        total_records = len(df)
        missing_data = df.isnull().sum().sum()
        missing_pct = (missing_data / (total_records * len(df.columns))) * 100
        
        # Duplicate records
        duplicates = df.duplicated().sum()
        duplicate_pct = (duplicates / total_records) * 100
        
        # Use Ollama for specialized quality assessment
        quality_summary = {
            'total_records': total_records,
            'missing_data_pct': missing_pct,
            'duplicate_pct': duplicate_pct,
            'columns_with_issues': df.columns[df.isnull().any()].tolist()
        }
        
        prompt = f"""
        Assess the data quality of this {data_type} dataset for actuarial use:
        
        QUALITY METRICS:
        - Total records: {total_records:,}
        - Missing data: {missing_pct:.1f}%
        - Duplicate records: {duplicate_pct:.1f}%
        - Columns with missing values: {quality_summary['columns_with_issues']}
        
        Provide assessment in JSON format:
        {{
            "overall_score": 85,
            "completeness_score": 90,
            "consistency_score": 80,
            "accuracy_score": 85,
            "timeliness_score": 90,
            "critical_issues": ["issue1", "issue2"],
            "warnings": ["warning1", "warning2"],
            "recommendations": ["rec1", "rec2"],
            "usability_for_pricing": "High|Medium|Low"
        }}
        """
        
        result = self.ollama.generate(prompt, self.actuarial_context)
        
        quality_assessment = {
            "overall_score": max(0, 100 - missing_pct * 2 - duplicate_pct * 3),
            "completeness_score": max(0, 100 - missing_pct * 3),
            "consistency_score": 85,  # Default
            "accuracy_score": 80,     # Default
            "timeliness_score": 90,   # Default
            "critical_issues": [],
            "warnings": [],
            "recommendations": [],
            "usability_for_pricing": "Medium"
        }
        
        if result['success']:
            try:
                response_text = result['response']
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = response_text[start:end]
                    ollama_assessment = json.loads(json_str)
                    quality_assessment.update(ollama_assessment)
            except Exception as e:
                self.logger.error(f"Could not parse quality assessment: {e}")
        
        return quality_assessment
    
    def process_uploaded_file(self, file_path: str, file_content: bytes = None) -> DataProcessingResult:
        """Complete processing pipeline for uploaded file"""
        
        try:
            self.logger.info(f"🔍 Processing uploaded file: {file_path}")
            
            # Step 1: Read file intelligently
            df = self.read_file_intelligently(file_path, file_content)
            self.logger.info(f"✅ Successfully loaded {len(df):,} records with {len(df.columns)} columns")
            
            # Step 2: Analyze data structure
            analysis = self.analyze_data_structure(df)
            data_type = analysis.get('data_type', 'unknown')
            self.logger.info(f"🎯 Identified as: {data_type} (confidence: {analysis.get('confidence', 0):.0%})")
            
            # Step 3: Standardize data
            standardized_df = self.standardize_data(df, data_type)
            self.logger.info(f"🔧 Data standardized")
            
            # Step 4: Assess quality
            quality = self.assess_data_quality(standardized_df, data_type)
            self.logger.info(f"📊 Quality score: {quality.get('overall_score', 0):.0f}/100")
            
            # Step 5: Store in database
            self._store_processed_data(standardized_df, data_type, file_path)
            
            return DataProcessingResult(
                success=True,
                data_type=data_type,
                standardized_data=standardized_df,
                quality_score=quality.get('overall_score', 0),
                issues=quality.get('critical_issues', []) + quality.get('warnings', []),
                recommendations=quality.get('recommendations', []),
                metadata={
                    'original_shape': df.shape,
                    'processed_shape': standardized_df.shape,
                    'analysis': analysis,
                    'quality_details': quality
                }
            )
            
        except Exception as e:
            self.logger.error(f"❌ Processing failed: {e}")
            return DataProcessingResult(
                success=False,
                data_type='unknown',
                standardized_data=None,
                quality_score=0,
                issues=[str(e)],
                recommendations=['Check file format and content'],
                metadata={'error': str(e)}
            )
    
    def _store_processed_data(self, df: pd.DataFrame, data_type: str, source_file: str):
        """Store processed data in local database"""
        
        db_path = "processed_data.db"
        
        with sqlite3.connect(db_path) as conn:
            # Create metadata table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS uploaded_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT,
                    data_type TEXT,
                    upload_date TEXT,
                    record_count INTEGER,
                    column_count INTEGER,
                    quality_score REAL
                )
            """)
            
            # Store metadata
            conn.execute("""
                INSERT INTO uploaded_files 
                (filename, data_type, upload_date, record_count, column_count, quality_score)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                Path(source_file).name,
                data_type,
                datetime.now().isoformat(),
                len(df),
                len(df.columns),
                0  # Will be updated with actual quality score
            ))
            
            # Store the actual data
            table_name = f"data_{data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            
            self.logger.info(f"💾 Data stored in database table: {table_name}")

def test_ollama_connection():
    """Test connection to Ollama"""
    
    client = OllamaClient()
    result = client.generate("Hello, are you working?")
    
    if result['success']:
        print("✅ Ollama connection successful!")
        print(f"Response: {result['response'][:100]}...")
        return True
    else:
        print(f"❌ Ollama connection failed: {result.get('error')}")
        return False

def main():
    """Test the data processor"""
    
    print("🧠 Testing Ollama Data Processor")
    print("=" * 40)
    
    # Test Ollama connection
    if not test_ollama_connection():
        print("Please start Ollama first: ollama serve")
        return
    
    # Test data processing
    processor = IntelligentDataProcessor()
    
    # Create sample CSV file
    sample_data = {
        'policy_num': ['POL001', 'POL002', 'POL003'],
        'age': [35, 42, 28],
        'sex': ['M', 'F', 'Male'],  # Inconsistent format
        'smoker': ['Y', 'No', '1'],  # Various formats
        'face_amt': [100000, 250000, 150000],
        'issue_dt': ['2020-01-01', '1/15/2020', '2021-03-15']  # Different date formats
    }
    
    sample_df = pd.DataFrame(sample_data)
    sample_file = "test_sample.csv"
    sample_df.to_csv(sample_file, index=False)
    
    # Process the sample file
    result = processor.process_uploaded_file(sample_file)
    
    if result.success:
        print(f"✅ Processing successful!")
        print(f"Data type: {result.data_type}")
        print(f"Quality score: {result.quality_score:.0f}/100")
        print(f"Records processed: {len(result.standardized_data):,}")
        
        if result.issues:
            print(f"Issues found: {result.issues}")
        if result.recommendations:
            print(f"Recommendations: {result.recommendations}")
            
    else:
        print(f"❌ Processing failed: {result.issues}")

if __name__ == "__main__":
    main()