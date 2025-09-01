"""
Statistical Content Analyzer - Phase 2 Implementation
Analyzes row content features and proximity patterns for junk detection
"""

import polars as pl
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from enum import Enum
import re
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

@dataclass
class StatisticalConfig:
    """Configuration for statistical analysis"""
    content_weight: float = 0.4
    proximity_weight: float = 0.3
    structural_weight: float = 0.3
    outlier_threshold: float = 2.0
    min_content_similarity: float = 0.7
    analyze_data_types: bool = True
    analyze_patterns: bool = True
    analyze_distributions: bool = True

@dataclass
class ContentFeatures:
    """Content features for a row"""
    row_index: int
    non_null_count: int
    null_percentage: float
    text_content_length: int
    numeric_fields_count: int
    date_fields_count: int
    special_chars_count: int
    capitalization_score: float
    whitespace_ratio: float
    unique_tokens: int
    repeated_patterns: int
    contains_keywords: List[str]
    data_type_consistency: float

@dataclass
class ProximityFeatures:
    """Proximity-based features"""
    row_index: int
    similarity_to_neighbors: float
    local_density: float
    pattern_consistency: float
    structural_similarity: float
    context_coherence: float

@dataclass
class StatisticalResult:
    """Results from statistical analysis"""
    junk_row_indices: List[int]
    content_scores: np.ndarray
    proximity_scores: np.ndarray
    combined_scores: np.ndarray
    outlier_threshold: float
    feature_importance: Dict[str, float]
    processing_time: float

class ContentAnalyzer:
    """Analyzes individual row content features"""
    
    def __init__(self, config: StatisticalConfig):
        self.config = config
        
        # Enhanced junk patterns with contextual understanding
        self.junk_keywords = {
            'summary': ['total:', 'sum:', 'grand total', 'subtotal', 'total', 'sum', 'summary', 'totals', 
                       'final total', 'grand sum', 'aggregate', 'count', 'records'],
            'footers': ['end of report', 'report footer', 'page footer', 'report generated', 'end of data',
                       'footer', 'generated', 'exported', 'created', 'system', 'end', 'final'],
            'metadata': ['generated on', 'exported on', 'created by', 'system generated', 'timestamp',
                        'processed by', 'report date', 'creation date', 'modified'],
            'empty_indicators': ['n/a', 'null', 'none', 'empty', '---', '===', '...', 'blank'],
            'measurement_units': ['products', 'items', 'rows', 'entries', 'records', 'total items']
        }
        
        # Context-aware patterns (combinations that suggest junk)
        self.contextual_junk_patterns = [
            # Pattern: number + unit (like "3 products", "450 total")
            (r'\d+\s+(products|items|rows|records|entries|total)', 0.8),
            # Pattern: summary phrases
            (r'(total|final|grand)\s+(summary|total|sum|count)', 0.9),
            # Pattern: system/report metadata
            (r'(generated|created|exported|processed)\s+(on|by)', 0.9),
            # Pattern: end/footer indicators
            (r'(end\s+of|report\s+footer|page\s+footer)', 0.9)
        ]
        
    def analyze_content_features(self, df: pl.DataFrame) -> List[ContentFeatures]:
        """Analyze content features for all rows"""
        
        features = []
        
        for i in range(df.shape[0]):
            row = df.row(i)
            
            # Basic content metrics
            non_null_count = sum(1 for val in row if val is not None and str(val).strip())
            null_percentage = 1.0 - (non_null_count / len(row))
            
            # Text analysis
            text_content = ' '.join(str(val) for val in row if val is not None)
            text_length = len(text_content)
            
            # Data type analysis
            numeric_count = self._count_numeric_fields(row)
            date_count = self._count_date_fields(row)
            
            # Pattern analysis
            special_chars = len(re.findall(r'[^\w\s]', text_content))
            caps_ratio = sum(1 for c in text_content if c.isupper()) / max(len(text_content), 1)
            whitespace_ratio = text_content.count(' ') / max(len(text_content), 1)
            
            # Semantic analysis
            tokens = set(text_content.lower().split())
            repeated_patterns = self._count_repeated_patterns(text_content)
            keywords = self._find_junk_keywords(text_content.lower(), i)
            
            # Data type consistency
            type_consistency = self._calculate_type_consistency(row, df.columns)
            
            features.append(ContentFeatures(
                row_index=i,
                non_null_count=non_null_count,
                null_percentage=null_percentage,
                text_content_length=text_length,
                numeric_fields_count=numeric_count,
                date_fields_count=date_count,
                special_chars_count=special_chars,
                capitalization_score=caps_ratio,
                whitespace_ratio=whitespace_ratio,
                unique_tokens=len(tokens),
                repeated_patterns=repeated_patterns,
                contains_keywords=keywords,
                data_type_consistency=type_consistency
            ))
        
        return features
    
    def _count_numeric_fields(self, row: List[Any]) -> int:
        """Count numeric fields in row"""
        count = 0
        for val in row:
            if val is None:
                continue
            try:
                # Try to convert to float
                float(str(val).replace(',', '').replace('$', '').strip())
                count += 1
            except (ValueError, AttributeError):
                continue
        return count
    
    def _count_date_fields(self, row: List[Any]) -> int:
        """Count date-like fields in row"""
        count = 0
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY
            r'\d{2,4}-\d{1,2}-\d{1,2}',  # YYYY-MM-DD
            r'\d{1,2}-\w{3}-\d{2,4}',    # DD-MMM-YYYY
        ]
        
        for val in row:
            if val is None:
                continue
            val_str = str(val).strip()
            
            for pattern in date_patterns:
                if re.search(pattern, val_str):
                    count += 1
                    break
        
        return count
    
    def _count_repeated_patterns(self, text: str) -> int:
        """Count repeated character patterns"""
        # Look for repeated characters like "---", "===", "..."
        patterns = re.findall(r'(.)\1{2,}', text)
        return len(patterns)
    
    def _find_junk_keywords(self, text_lower: str, row_index: int = None) -> List[str]:
        """Find junk-indicating keywords with contextual pattern matching"""
        import re
        found_keywords = []
        
        # Traditional keyword matching
        for category, keywords in self.junk_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_keywords.append(f"{category}:{keyword}")
        
        # Enhanced contextual pattern matching
        for pattern, confidence in self.contextual_junk_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                match_text = match if isinstance(match, str) else ' '.join(match)
                found_keywords.append(f"contextual_pattern:{match_text}:{confidence}")
        
        return found_keywords
    
    def _calculate_type_consistency(self, row: List[Any], columns: List[str]) -> float:
        """Calculate how consistent this row's data types are with column expectations"""
        
        # This is a simplified version - in practice, would analyze column patterns
        consistent_fields = 0
        total_fields = 0
        
        for val in row:
            if val is not None and str(val).strip():
                total_fields += 1
                # Simple heuristic: if it looks like data (not metadata), it's consistent
                val_str = str(val).strip()
                if not any(keyword in val_str.lower() for keywords in self.junk_keywords.values() for keyword in keywords):
                    consistent_fields += 1
        
        return consistent_fields / max(total_fields, 1)

class ProximityAnalyzer:
    """Analyzes proximity patterns and neighborhood similarity"""
    
    def __init__(self, config: StatisticalConfig):
        self.config = config
    
    def analyze_proximity_features(self, df: pl.DataFrame, 
                                 content_features: List[ContentFeatures]) -> List[ProximityFeatures]:
        """Analyze proximity-based features"""
        
        features = []
        
        for i in range(df.shape[0]):
            # Calculate neighborhood similarity
            neighbor_similarity = self._calculate_neighbor_similarity(i, content_features)
            
            # Calculate local density (how similar are nearby rows)
            local_density = self._calculate_local_density(i, df)
            
            # Pattern consistency with neighbors
            pattern_consistency = self._calculate_pattern_consistency(i, df)
            
            # Structural similarity
            structural_similarity = self._calculate_structural_similarity(i, df)
            
            # Context coherence
            context_coherence = self._calculate_context_coherence(i, df)
            
            features.append(ProximityFeatures(
                row_index=i,
                similarity_to_neighbors=neighbor_similarity,
                local_density=local_density,
                pattern_consistency=pattern_consistency,
                structural_similarity=structural_similarity,
                context_coherence=context_coherence
            ))
        
        return features
    
    def _calculate_neighbor_similarity(self, row_idx: int, 
                                     content_features: List[ContentFeatures],
                                     window_size: int = 3) -> float:
        """Calculate similarity to neighboring rows"""
        
        current_features = content_features[row_idx]
        neighbors = []
        
        # Get neighbors within window
        start = max(0, row_idx - window_size)
        end = min(len(content_features), row_idx + window_size + 1)
        
        for i in range(start, end):
            if i != row_idx:
                neighbors.append(content_features[i])
        
        if not neighbors:
            return 0.0
        
        # Calculate similarity based on key features
        similarities = []
        
        for neighbor in neighbors:
            similarity = 0.0
            
            # Null percentage similarity
            null_diff = abs(current_features.null_percentage - neighbor.null_percentage)
            similarity += (1.0 - null_diff)
            
            # Non-null count similarity  
            count_diff = abs(current_features.non_null_count - neighbor.non_null_count)
            max_count = max(current_features.non_null_count, neighbor.non_null_count, 1)
            similarity += (1.0 - count_diff / max_count)
            
            # Text length similarity
            length_diff = abs(current_features.text_content_length - neighbor.text_content_length)
            max_length = max(current_features.text_content_length, neighbor.text_content_length, 1)
            similarity += (1.0 - length_diff / max_length)
            
            # Data type consistency similarity
            type_diff = abs(current_features.data_type_consistency - neighbor.data_type_consistency)
            similarity += (1.0 - type_diff)
            
            similarities.append(similarity / 4.0)  # Average of 4 metrics
        
        return np.mean(similarities)
    
    def _calculate_local_density(self, row_idx: int, df: pl.DataFrame, 
                               window_size: int = 5) -> float:
        """Calculate local density of content"""
        
        start = max(0, row_idx - window_size)
        end = min(df.shape[0], row_idx + window_size + 1)
        
        local_rows = df[start:end]
        
        # Calculate average non-null density in neighborhood
        densities = []
        
        for i in range(local_rows.shape[0]):
            row = local_rows.row(i)
            non_null_count = sum(1 for val in row if val is not None and str(val).strip())
            density = non_null_count / len(row)
            densities.append(density)
        
        return np.mean(densities)
    
    def _calculate_pattern_consistency(self, row_idx: int, df: pl.DataFrame) -> float:
        """Calculate pattern consistency with surrounding rows"""
        
        current_row = df.row(row_idx)
        current_pattern = self._extract_row_pattern(current_row)
        
        # Compare with neighbors
        neighbors = []
        for offset in [-2, -1, 1, 2]:
            neighbor_idx = row_idx + offset
            if 0 <= neighbor_idx < df.shape[0]:
                neighbor_row = df.row(neighbor_idx)
                neighbor_pattern = self._extract_row_pattern(neighbor_row)
                neighbors.append(neighbor_pattern)
        
        if not neighbors:
            return 0.5  # Neutral score
        
        # Calculate pattern similarity
        similarities = []
        for neighbor_pattern in neighbors:
            similarity = self._compare_patterns(current_pattern, neighbor_pattern)
            similarities.append(similarity)
        
        return np.mean(similarities)
    
    def _extract_row_pattern(self, row: List[Any]) -> Dict[str, Any]:
        """Extract pattern characteristics from a row"""
        
        pattern = {
            'null_positions': [i for i, val in enumerate(row) if val is None or not str(val).strip()],
            'data_positions': [i for i, val in enumerate(row) if val is not None and str(val).strip()],
            'numeric_positions': [],
            'text_positions': [],
            'total_length': len(''.join(str(val) for val in row if val is not None))
        }
        
        for i, val in enumerate(row):
            if val is not None and str(val).strip():
                try:
                    float(str(val).replace(',', '').replace('$', '').strip())
                    pattern['numeric_positions'].append(i)
                except (ValueError, AttributeError):
                    pattern['text_positions'].append(i)
        
        return pattern
    
    def _compare_patterns(self, pattern1: Dict, pattern2: Dict) -> float:
        """Compare two row patterns"""
        
        similarity = 0.0
        
        # Compare null positions
        null1 = set(pattern1['null_positions'])
        null2 = set(pattern2['null_positions'])
        null_similarity = len(null1 & null2) / max(len(null1 | null2), 1)
        similarity += null_similarity
        
        # Compare data positions
        data1 = set(pattern1['data_positions'])
        data2 = set(pattern2['data_positions'])
        data_similarity = len(data1 & data2) / max(len(data1 | data2), 1)
        similarity += data_similarity
        
        # Compare total length
        length1 = pattern1['total_length']
        length2 = pattern2['total_length']
        length_similarity = 1.0 - abs(length1 - length2) / max(length1, length2, 1)
        similarity += length_similarity
        
        return similarity / 3.0  # Average of 3 metrics
    
    def _calculate_structural_similarity(self, row_idx: int, df: pl.DataFrame) -> float:
        """Calculate structural similarity within the dataset"""
        
        current_row = df.row(row_idx)
        current_structure = self._get_row_structure(current_row)
        
        # Compare with random sample of other rows
        sample_size = min(10, df.shape[0] - 1)
        sample_indices = np.random.choice([i for i in range(df.shape[0]) if i != row_idx], 
                                        min(sample_size, df.shape[0] - 1), replace=False)
        
        similarities = []
        for idx in sample_indices:
            other_row = df.row(idx)
            other_structure = self._get_row_structure(other_row)
            similarity = self._compare_structures(current_structure, other_structure)
            similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.5
    
    def _get_row_structure(self, row: List[Any]) -> Dict[str, Any]:
        """Get structural characteristics of a row"""
        
        structure = {
            'field_count': len([val for val in row if val is not None and str(val).strip()]),
            'empty_count': len([val for val in row if val is None or not str(val).strip()]),
            'avg_field_length': 0,
            'has_numeric': False,
            'has_text': False
        }
        
        field_lengths = []
        for val in row:
            if val is not None and str(val).strip():
                field_lengths.append(len(str(val)))
                
                # Check if numeric
                try:
                    float(str(val).replace(',', '').replace('$', '').strip())
                    structure['has_numeric'] = True
                except (ValueError, AttributeError):
                    structure['has_text'] = True
        
        structure['avg_field_length'] = np.mean(field_lengths) if field_lengths else 0
        
        return structure
    
    def _compare_structures(self, struct1: Dict, struct2: Dict) -> float:
        """Compare two structural profiles"""
        
        similarity = 0.0
        
        # Field count similarity
        count_diff = abs(struct1['field_count'] - struct2['field_count'])
        max_count = max(struct1['field_count'], struct2['field_count'], 1)
        similarity += (1.0 - count_diff / max_count)
        
        # Length similarity
        length_diff = abs(struct1['avg_field_length'] - struct2['avg_field_length'])
        max_length = max(struct1['avg_field_length'], struct2['avg_field_length'], 1)
        similarity += (1.0 - length_diff / max_length)
        
        # Data type similarity
        if struct1['has_numeric'] == struct2['has_numeric']:
            similarity += 0.5
        if struct1['has_text'] == struct2['has_text']:
            similarity += 0.5
        
        return similarity / 3.0
    
    def _calculate_context_coherence(self, row_idx: int, df: pl.DataFrame) -> float:
        """Calculate how well this row fits the overall data context with header awareness"""
        
        current_row = df.row(row_idx)
        coherence_score = 0.0
        
        # Analyze potential header characteristics
        header_score = self._analyze_header_potential(row_idx, df, current_row)
        
        # Analyze data characteristics  
        data_score = self._analyze_data_potential(row_idx, df, current_row)
        
        # Analyze junk characteristics
        junk_score = self._analyze_junk_potential(row_idx, df, current_row)
        
        # Context coherence is high if it's either a good header OR good data (but not junk)
        if header_score > 0.7:  # Strong header candidate
            coherence_score = header_score
        elif data_score > 0.6:  # Good data row
            coherence_score = data_score  
        else:  # Likely junk
            coherence_score = max(0, 1.0 - junk_score)
        
        return min(coherence_score, 1.0)
    
    def _analyze_header_potential(self, row_idx: int, df: pl.DataFrame, current_row) -> float:
        """Analyze if this row could be a legitimate column header"""
        
        score = 0.0
        text_content = ' '.join(str(val) for val in current_row if val is not None).lower()
        
        # Field count alignment with typical data rows
        non_null_count = sum(1 for val in current_row if val is not None and str(val).strip())
        if non_null_count >= len(current_row) * 0.6:  # Well-filled row
            score += 0.3
            
        # Check if positioned appropriately (before main data block)
        if self._is_positioned_like_header(row_idx, df):
            score += 0.3
            
        # Column semantic alignment - do fields make sense as column descriptors?
        if self._has_column_descriptor_pattern(current_row):
            score += 0.4
            
        return score
    
    def _analyze_data_potential(self, row_idx: int, df: pl.DataFrame, current_row) -> float:
        """Analyze if this row looks like normal data"""
        
        score = 0.0
        
        # Position in data region (not at very beginning or end)
        if row_idx >= 1 and row_idx < df.shape[0] - 2:
            score += 0.2
            
        # Consistent field count
        non_null_count = sum(1 for val in current_row if val is not None and str(val).strip())
        if non_null_count >= len(current_row) * 0.5:
            score += 0.3
            
        # Mixed data types (typical of data rows)
        if self._has_mixed_data_types(current_row):
            score += 0.3
            
        # No strong junk indicators
        text_content = ' '.join(str(val) for val in current_row if val is not None).lower()
        junk_patterns = ['total:', 'sum:', 'report:', 'generated on', 'end of']
        if not any(pattern in text_content for pattern in junk_patterns):
            score += 0.2
            
        return score
    
    def _analyze_junk_potential(self, row_idx: int, df: pl.DataFrame, current_row) -> float:
        """Analyze if this row looks like junk/metadata with enhanced footer detection"""
        
        score = 0.0
        text_content = ' '.join(str(val) for val in current_row if val is not None).lower()
        
        # High null percentage
        null_count = sum(1 for val in current_row if val is None or not str(val).strip())
        if null_count > len(current_row) * 0.7:
            score += 0.4
            
        # Enhanced footer detection - check position
        position_ratio = row_idx / df.shape[0]
        if position_ratio > 0.8:  # Last 20% of data
            # More aggressive junk detection for footer area
            footer_patterns = ['total', 'sum', 'summary', 'end', 'footer', 'generated', 'report']
            footer_indicators = sum(1 for pattern in footer_patterns if pattern in text_content)
            if footer_indicators >= 1:
                score += 0.6  # High score for footer area with junk words
                
        # Strong junk patterns
        strong_junk = ['total:', 'sum:', 'grand total', 'report generated', 'end of report', 'summary', 'totals']
        if any(pattern in text_content for pattern in strong_junk):
            score += 0.5
            
        # Metadata patterns  
        metadata_patterns = ['generated on', 'exported on', 'created by', 'system']
        if any(pattern in text_content for pattern in metadata_patterns):
            score += 0.3
            
        # Check if row contains mostly repeated characters (separator lines)
        if len(text_content) > 0:
            unique_chars = set(text_content.replace(' ', ''))
            if len(unique_chars) <= 3 and len(text_content) > 5:  # Like "---" or "==="
                score += 0.7
                
        # Single field dominance (often summary rows)
        non_null_values = [str(val).strip() for val in current_row if val is not None and str(val).strip()]
        if len(non_null_values) == 1 and any(word in non_null_values[0].lower() for word in ['total', 'sum', 'summary']):
            score += 0.6
            
        return min(score, 1.0)
    
    def _is_positioned_like_header(self, row_idx: int, df: pl.DataFrame) -> bool:
        """Check if row is positioned where headers typically appear"""
        
        # Headers can appear anywhere, but usually before the main data block
        # Look for rows that are followed by consistent data patterns
        
        if row_idx >= df.shape[0] - 2:  # Too close to end
            return False
            
        # Check if followed by rows with similar structure (data pattern)
        consistent_followers = 0
        for i in range(row_idx + 1, min(row_idx + 5, df.shape[0])):
            follower_row = df.row(i)
            follower_non_null = sum(1 for val in follower_row if val is not None and str(val).strip())
            current_non_null = sum(1 for val in df.row(row_idx) if val is not None and str(val).strip())
            
            # Similar field counts suggest this could be header + data pattern
            if abs(follower_non_null - current_non_null) <= 2:
                consistent_followers += 1
                
        return consistent_followers >= 2
    
    def _has_column_descriptor_pattern(self, current_row) -> bool:
        """Check if row fields look like column descriptors"""
        
        # Column descriptors typically:
        # 1. Are short text (not long sentences)
        # 2. Don't contain numbers/dates as primary content
        # 3. Have descriptor-like words
        
        descriptor_count = 0
        total_fields = 0
        
        for val in current_row:
            if val is not None and str(val).strip():
                total_fields += 1
                val_str = str(val).strip()
                
                # Short, descriptive text (not sentences or long content)
                if len(val_str) < 30 and '_' in val_str or val_str.replace('_', ' ').replace('-', ' ').count(' ') <= 2:
                    descriptor_count += 1
                    
        return descriptor_count >= total_fields * 0.6 if total_fields > 0 else False
    
    def _has_mixed_data_types(self, current_row) -> bool:
        """Check if row has mixed data types typical of data rows"""
        
        types = {'numeric': 0, 'text': 0, 'date': 0}
        
        for val in current_row:
            if val is not None and str(val).strip():
                val_str = str(val).strip()
                
                # Simple type detection
                if val_str.replace('.', '').replace('-', '').isdigit():
                    types['numeric'] += 1
                elif '/' in val_str or '-' in val_str and len(val_str) >= 8:
                    types['date'] += 1  
                else:
                    types['text'] += 1
        
        # Good data rows typically have at least 2 different types
        return sum(1 for count in types.values() if count > 0) >= 2

class StatisticalJunkDetector:
    """
    Statistical junk detector using content and proximity analysis
    Phase 2 of the hybrid approach
    """
    
    def __init__(self, config: StatisticalConfig = None):
        self.config = config or StatisticalConfig()
        self.content_analyzer = ContentAnalyzer(self.config)
        self.proximity_analyzer = ProximityAnalyzer(self.config)
    
    def detect_junk_rows(self, df: pl.DataFrame, 
                        return_detailed: bool = False) -> Union[List[int], StatisticalResult]:
        """
        Detect junk rows using statistical analysis
        
        Args:
            df: Input dataframe
            return_detailed: Return detailed StatisticalResult if True
            
        Returns:
            List of junk row indices or detailed StatisticalResult
        """
        import time
        start_time = time.time()
        
        logger.info(f"Starting statistical junk detection on {df.shape[0]} rows")
        
        # Step 1: Analyze content features
        content_features = self.content_analyzer.analyze_content_features(df)
        
        # Step 2: Analyze proximity features
        proximity_features = self.proximity_analyzer.analyze_proximity_features(df, content_features)
        
        # Step 3: Calculate combined scores
        content_scores = self._calculate_content_scores(content_features)
        proximity_scores = self._calculate_proximity_scores(proximity_features)
        combined_scores = self._combine_scores(content_scores, proximity_scores)
        
        # Step 4: Determine outlier threshold and identify junk
        threshold = self._calculate_outlier_threshold(combined_scores)
        junk_indices = np.where(combined_scores > threshold)[0].tolist()
        
        # Step 5: Calculate feature importance
        feature_importance = self._calculate_feature_importance(content_features, proximity_features, junk_indices)
        
        processing_time = time.time() - start_time
        logger.info(f"Statistical detection completed in {processing_time:.2f}s, found {len(junk_indices)} junk rows")
        
        result = StatisticalResult(
            junk_row_indices=junk_indices,
            content_scores=content_scores,
            proximity_scores=proximity_scores,
            combined_scores=combined_scores,
            outlier_threshold=threshold,
            feature_importance=feature_importance,
            processing_time=processing_time
        )
        
        if return_detailed:
            return result
        else:
            return result.junk_row_indices
    
    def _calculate_content_scores(self, content_features: List[ContentFeatures]) -> np.ndarray:
        """Calculate outlier scores based on content features"""
        
        scores = np.zeros(len(content_features))
        
        for i, features in enumerate(content_features):
            score = 0.0
            
            # High null percentage indicates junk
            score += features.null_percentage * 0.3
            
            # Very short or very long text content
            avg_length = np.mean([f.text_content_length for f in content_features])
            std_length = np.std([f.text_content_length for f in content_features])
            
            if std_length > 0:
                length_z_score = abs(features.text_content_length - avg_length) / std_length
                score += min(length_z_score / 3.0, 0.3)  # Cap at 0.3
            
            # Enhanced keyword scoring with contextual patterns
            if features.contains_keywords:
                keyword_score = 0.0
                for keyword in features.contains_keywords:
                    if 'contextual_pattern:' in keyword:
                        # Extract confidence from contextual pattern
                        confidence = float(keyword.split(':')[-1])
                        keyword_score += confidence * 0.6  # High weight for contextual patterns
                    else:
                        keyword_score += 0.2  # Standard weight for simple keywords
                
                score += min(keyword_score, 0.8)  # Cap total keyword score
            
            # Low data type consistency
            score += (1.0 - features.data_type_consistency) * 0.2
            
            # Unusual capitalization patterns
            if features.capitalization_score > 0.8 or features.capitalization_score < 0.1:
                score += 0.1
            
            scores[i] = score
        
        return scores
    
    def _calculate_proximity_scores(self, proximity_features: List[ProximityFeatures]) -> np.ndarray:
        """Calculate outlier scores based on proximity features"""
        
        scores = np.zeros(len(proximity_features))
        
        for i, features in enumerate(proximity_features):
            score = 0.0
            
            # Low similarity to neighbors
            score += (1.0 - features.similarity_to_neighbors) * 0.4
            
            # Low pattern consistency
            score += (1.0 - features.pattern_consistency) * 0.3
            
            # Low context coherence
            score += (1.0 - features.context_coherence) * 0.3
            
            scores[i] = score
        
        return scores
    
    def _combine_scores(self, content_scores: np.ndarray, proximity_scores: np.ndarray) -> np.ndarray:
        """Combine content and proximity scores"""
        
        combined = (content_scores * self.config.content_weight + 
                   proximity_scores * self.config.proximity_weight)
        
        # Normalize to [0, 1] range
        if np.max(combined) > 0:
            combined = combined / np.max(combined)
        
        return combined
    

    def _calculate_outlier_threshold(self, scores: np.ndarray) -> float:
        """Calculate dynamic outlier threshold"""
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        threshold = mean_score + (self.config.outlier_threshold * std_score)
        
        # Ensure reasonable bounds - made more aggressive for better junk detection
        threshold = max(0.3, min(0.65, threshold))  # Changed from 0.7 to 0.65 to catch more junk
        
        return threshold
    
    def _calculate_feature_importance(self, content_features: List[ContentFeatures], 
                                    proximity_features: List[ProximityFeatures],
                                    junk_indices: List[int]) -> Dict[str, float]:
        """Calculate importance of different features for junk detection"""
        
        if not junk_indices:
            return {}
        
        importance = {
            'null_percentage': 0.0,
            'text_length': 0.0,
            'keywords': 0.0,
            'neighbor_similarity': 0.0,
            'pattern_consistency': 0.0,
            'context_coherence': 0.0
        }
        
        # Calculate average feature values for junk vs non-junk rows
        junk_set = set(junk_indices)
        
        junk_null_pct = np.mean([content_features[i].null_percentage for i in junk_indices])
        clean_null_pct = np.mean([content_features[i].null_percentage for i in range(len(content_features)) if i not in junk_set])
        importance['null_percentage'] = abs(junk_null_pct - clean_null_pct)
        
        junk_keywords = np.mean([len(content_features[i].contains_keywords) for i in junk_indices])
        clean_keywords = np.mean([len(content_features[i].contains_keywords) for i in range(len(content_features)) if i not in junk_set])
        importance['keywords'] = abs(junk_keywords - clean_keywords)
        
        junk_similarity = np.mean([proximity_features[i].similarity_to_neighbors for i in junk_indices])
        clean_similarity = np.mean([proximity_features[i].similarity_to_neighbors for i in range(len(proximity_features)) if i not in junk_set])
        importance['neighbor_similarity'] = abs(clean_similarity - junk_similarity)  # Higher for clean is better
        
        # Normalize importance scores
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        return importance

# Convenience functions
def create_statistical_detector(content_weight: float = 0.4,
                              proximity_weight: float = 0.3,
                              outlier_threshold: float = 2.0) -> StatisticalJunkDetector:
    """Create a configured statistical detector"""
    
    config = StatisticalConfig(
        content_weight=content_weight,
        proximity_weight=proximity_weight,
        structural_weight=1.0 - content_weight - proximity_weight,
        outlier_threshold=outlier_threshold
    )
    
    return StatisticalJunkDetector(config)

def quick_statistical_clean(df: pl.DataFrame) -> Tuple[pl.DataFrame, List[int]]:
    """Quick statistical cleaning with default settings"""
    
    detector = create_statistical_detector()
    junk_indices = detector.detect_junk_rows(df)
    
    # Remove junk rows
    if junk_indices:
        mask = pl.Series(range(df.shape[0])).is_in(junk_indices).not_()
        clean_df = df.filter(mask)
    else:
        clean_df = df
    
    return clean_df, junk_indices