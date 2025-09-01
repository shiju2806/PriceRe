"""
Universal Data Source Adapters
Supports: Files, Databases, APIs, Streaming data, Cloud storage
"""

import pandas as pd
import polars as pl
from abc import ABC, abstractmethod
from typing import Dict, Any, Iterator, Optional, Union, List
from pathlib import Path
import io
import sqlite3
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class DataSourceType(Enum):
    """Supported data source types"""
    FILE_CSV = "csv"
    FILE_EXCEL = "excel" 
    FILE_JSON = "json"
    FILE_PARQUET = "parquet"
    DATABASE_SQLITE = "sqlite"
    DATABASE_POSTGRES = "postgres"
    DATABASE_MYSQL = "mysql"
    API_REST = "rest_api"
    STREAM_KAFKA = "kafka"
    CLOUD_S3 = "s3"
    CLOUD_GCS = "gcs"
    MEMORY_DATAFRAME = "dataframe"

@dataclass
class DataSourceConfig:
    """Configuration for data sources"""
    source_type: DataSourceType
    connection_string: Optional[str] = None
    table_name: Optional[str] = None
    query: Optional[str] = None
    batch_size: int = 1000
    chunk_size: int = 10000
    headers: Optional[Dict[str, str]] = None
    credentials: Optional[Dict[str, str]] = None
    
class UniversalDataSource(ABC):
    """Abstract base class for all data sources"""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.metadata = {}
        
    @abstractmethod
    def read_data(self, limit: Optional[int] = None) -> pl.DataFrame:
        """Read data from source"""
        pass
        
    @abstractmethod
    def read_streaming(self, chunk_size: int = 1000) -> Iterator[pl.DataFrame]:
        """Stream data in chunks"""
        pass
        
    @abstractmethod
    def get_schema(self) -> Dict[str, str]:
        """Get column schema"""
        pass
        
    @abstractmethod
    def get_row_count(self) -> int:
        """Get total row count"""
        pass

class FileDataSource(UniversalDataSource):
    """Handle various file formats"""
    
    def __init__(self, file_path: Union[str, Path, io.BytesIO], source_type: DataSourceType = None):
        if isinstance(file_path, (str, Path)):
            file_path = Path(file_path)
            if source_type is None:
                # Auto-detect from extension
                ext = file_path.suffix.lower()
                type_map = {
                    '.csv': DataSourceType.FILE_CSV,
                    '.xlsx': DataSourceType.FILE_EXCEL,
                    '.xls': DataSourceType.FILE_EXCEL,
                    '.json': DataSourceType.FILE_JSON,
                    '.parquet': DataSourceType.FILE_PARQUET
                }
                source_type = type_map.get(ext, DataSourceType.FILE_CSV)
        
        config = DataSourceConfig(source_type=source_type)
        super().__init__(config)
        self.file_path = file_path
        
    def read_data(self, limit: Optional[int] = None) -> pl.DataFrame:
        """Read file data with format detection"""
        try:
            if self.config.source_type == DataSourceType.FILE_CSV:
                df = pl.read_csv(self.file_path, ignore_errors=True)
            elif self.config.source_type == DataSourceType.FILE_EXCEL:
                # Use pandas for Excel, convert to polars with string dtypes to avoid conversion errors
                df_pandas = pd.read_excel(self.file_path, dtype=str)
                df = pl.from_pandas(df_pandas)
            elif self.config.source_type == DataSourceType.FILE_JSON:
                df = pl.read_json(self.file_path)
            elif self.config.source_type == DataSourceType.FILE_PARQUET:
                df = pl.read_parquet(self.file_path)
            else:
                raise ValueError(f"Unsupported file type: {self.config.source_type}")
                
            if limit:
                df = df.head(limit)
                
            logger.info(f"Loaded {df.shape[0]} rows from {self.file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to read {self.file_path}: {e}")
            raise
    
    def read_streaming(self, chunk_size: int = 1000) -> Iterator[pl.DataFrame]:
        """Stream file in chunks"""
        if self.config.source_type == DataSourceType.FILE_CSV:
            # Use lazy reading for large CSV files
            df_lazy = pl.scan_csv(self.file_path, ignore_errors=True)
            
            total_rows = df_lazy.select(pl.count()).collect().item()
            
            for start in range(0, total_rows, chunk_size):
                chunk = df_lazy.slice(start, chunk_size).collect()
                yield chunk
        else:
            # For other formats, read all then chunk
            df = self.read_data()
            for start in range(0, len(df), chunk_size):
                yield df[start:start + chunk_size]
    
    def get_schema(self) -> Dict[str, str]:
        """Get column types"""
        df = self.read_data(limit=100)  # Sample for schema
        return dict(zip(df.columns, [str(dtype) for dtype in df.dtypes]))
    
    def get_row_count(self) -> int:
        """Get total row count efficiently"""
        if self.config.source_type == DataSourceType.FILE_CSV:
            df_lazy = pl.scan_csv(self.file_path, ignore_errors=True)
            return df_lazy.select(pl.count()).collect().item()
        else:
            return len(self.read_data())

class DatabaseDataSource(UniversalDataSource):
    """Handle database connections"""
    
    def __init__(self, connection_string: str, table_name: str = None, query: str = None):
        # Auto-detect database type from connection string
        if connection_string.startswith('sqlite'):
            source_type = DataSourceType.DATABASE_SQLITE
        elif connection_string.startswith('postgresql'):
            source_type = DataSourceType.DATABASE_POSTGRES
        elif connection_string.startswith('mysql'):
            source_type = DataSourceType.DATABASE_MYSQL
        else:
            source_type = DataSourceType.DATABASE_SQLITE  # Default
            
        config = DataSourceConfig(
            source_type=source_type,
            connection_string=connection_string,
            table_name=table_name,
            query=query
        )
        super().__init__(config)
        
    def read_data(self, limit: Optional[int] = None) -> pl.DataFrame:
        """Read from database"""
        if self.config.source_type == DataSourceType.DATABASE_SQLITE:
            conn = sqlite3.connect(self.config.connection_string)
            
            if self.config.query:
                query = self.config.query
            else:
                query = f"SELECT * FROM {self.config.table_name}"
                
            if limit:
                query += f" LIMIT {limit}"
                
            df = pl.read_database(query, connection=conn)
            conn.close()
            return df
        else:
            # For PostgreSQL/MySQL, would use appropriate drivers
            raise NotImplementedError(f"Database type {self.config.source_type} not yet implemented")
    
    def read_streaming(self, chunk_size: int = 1000) -> Iterator[pl.DataFrame]:
        """Stream database results"""
        offset = 0
        while True:
            if self.config.query:
                query = f"{self.config.query} LIMIT {chunk_size} OFFSET {offset}"
            else:
                query = f"SELECT * FROM {self.config.table_name} LIMIT {chunk_size} OFFSET {offset}"
                
            chunk = self.read_data()  # Would need to modify for chunking
            if len(chunk) == 0:
                break
                
            yield chunk
            offset += chunk_size
    
    def get_schema(self) -> Dict[str, str]:
        """Get table schema"""
        sample = self.read_data(limit=1)
        return dict(zip(sample.columns, [str(dtype) for dtype in sample.dtypes]))
    
    def get_row_count(self) -> int:
        """Get total row count"""
        if self.config.query:
            count_query = f"SELECT COUNT(*) FROM ({self.config.query}) subquery"
        else:
            count_query = f"SELECT COUNT(*) FROM {self.config.table_name}"
            
        conn = sqlite3.connect(self.config.connection_string)
        result = conn.execute(count_query).fetchone()[0]
        conn.close()
        return result

class DataFrameDataSource(UniversalDataSource):
    """Handle in-memory DataFrames"""
    
    def __init__(self, dataframe: Union[pd.DataFrame, pl.DataFrame]):
        config = DataSourceConfig(source_type=DataSourceType.MEMORY_DATAFRAME)
        super().__init__(config)
        
        # Convert to Polars if needed
        if isinstance(dataframe, pd.DataFrame):
            self.df = pl.from_pandas(dataframe)
        else:
            self.df = dataframe
    
    def read_data(self, limit: Optional[int] = None) -> pl.DataFrame:
        if limit:
            return self.df.head(limit)
        return self.df
    
    def read_streaming(self, chunk_size: int = 1000) -> Iterator[pl.DataFrame]:
        for start in range(0, len(self.df), chunk_size):
            yield self.df[start:start + chunk_size]
    
    def get_schema(self) -> Dict[str, str]:
        return dict(zip(self.df.columns, [str(dtype) for dtype in self.df.dtypes]))
    
    def get_row_count(self) -> int:
        return len(self.df)

class UniversalDataSourceFactory:
    """Factory to create appropriate data source"""
    
    @staticmethod
    def create_source(source: Union[str, Path, pd.DataFrame, pl.DataFrame, io.BytesIO]) -> UniversalDataSource:
        """Create appropriate data source from input"""
        
        if isinstance(source, (pd.DataFrame, pl.DataFrame)):
            return DataFrameDataSource(source)
        
        elif isinstance(source, (str, Path)):
            source_path = Path(source)
            
            # Check if it's a database connection string
            if str(source).startswith(('sqlite:', 'postgresql:', 'mysql:')):
                return DatabaseDataSource(str(source))
            
            # Check if it's a file
            elif source_path.exists() or isinstance(source, io.BytesIO):
                return FileDataSource(source)
            
            else:
                raise ValueError(f"Cannot determine data source type for: {source}")
        
        elif isinstance(source, io.BytesIO):
            return FileDataSource(source)
        
        else:
            raise TypeError(f"Unsupported data source type: {type(source)}")

# Convenience functions
def read_any_source(source: Union[str, Path, pd.DataFrame, pl.DataFrame, io.BytesIO], 
                   limit: Optional[int] = None) -> pl.DataFrame:
    """Read any data source into Polars DataFrame"""
    data_source = UniversalDataSourceFactory.create_source(source)
    return data_source.read_data(limit=limit)

def stream_any_source(source: Union[str, Path, pd.DataFrame, pl.DataFrame], 
                     chunk_size: int = 1000) -> Iterator[pl.DataFrame]:
    """Stream any data source"""
    data_source = UniversalDataSourceFactory.create_source(source)
    return data_source.read_streaming(chunk_size=chunk_size)