"""
Smart Header Row Detection
Finds the real column headers in messy datasets before cleaning
"""

import polars as pl
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class HeaderCandidate:
    """A potential header row candidate"""
    row_index: int
    confidence_score: float
    field_count: int
    descriptor_score: float
    position_score: float
    data_alignment_score: float
    explanation: str

@dataclass 
class HeaderDetectionResult:
    """Result of header detection"""
    header_row_index: int
    confidence: float
    candidates: List[HeaderCandidate]
    rows_to_remove_above: List[int]
    column_names: List[str]
    processing_notes: List[str]

class SmartHeaderDetector:
    """
    Detects the real header row using statistical and proximity analysis
    """
    
    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = min_confidence
    
    def detect_header_row(self, df: pl.DataFrame) -> HeaderDetectionResult:
        """
        Detect the most likely header row in the dataset
        
        Args:
            df: Raw dataframe (no assumptions about headers)
            
        Returns:
            HeaderDetectionResult with header row index and confidence
        """
        
        logger.info(f"Starting header detection on {df.shape[0]} rows")
        
        # Step 1: Analyze all rows as potential headers
        candidates = self._analyze_all_candidates(df)
        
        # Step 2: Score and rank candidates
        scored_candidates = self._score_candidates(df, candidates)
        
        # Step 3: Select best candidate
        best_candidate = max(scored_candidates, key=lambda x: x.confidence_score)
        
        # Step 4: Validate the selection
        if best_candidate.confidence_score < self.min_confidence:
            logger.warning(f"Low confidence header detection: {best_candidate.confidence_score}")
        
        # Step 5: Build result
        result = self._build_result(df, best_candidate, scored_candidates)
        
        logger.info(f"Header detected at row {result.header_row_index} with confidence {result.confidence:.3f}")
        
        return result
    
    def _analyze_all_candidates(self, df: pl.DataFrame) -> List[Dict]:
        """Analyze each row as a potential header"""
        
        candidates = []
        
        # Only consider first 80% of rows as potential headers
        max_header_row = int(df.shape[0] * 0.8)
        
        for i in range(min(max_header_row, df.shape[0])):
            try:
                row = df.row(i)
                
                # Basic metrics
                field_count = sum(1 for val in row if val is not None and str(val).strip())
                
                # Skip if too few fields
                if field_count < 2:
                    continue
                
                candidate = {
                    'row_index': i,
                    'row_data': row,
                    'field_count': field_count,
                    'text_content': ' '.join(str(val) for val in row if val is not None)
                }
                
                candidates.append(candidate)
                
            except Exception as e:
                logger.warning(f"Error analyzing row {i}: {e}")
                continue
        
        return candidates
    
    def _score_candidates(self, df: pl.DataFrame, candidates: List[Dict]) -> List[HeaderCandidate]:
        """Score each candidate based on multiple factors"""
        
        scored_candidates = []
        
        for candidate in candidates:
            row_idx = candidate['row_index']
            row_data = candidate['row_data']
            
            # Factor 1: Descriptor Pattern Score (are these column-like names?)
            descriptor_score = self._calculate_descriptor_score(row_data)
            
            # Factor 2: Position Score (headers usually not at very end)
            position_score = self._calculate_position_score(row_idx, df.shape[0])
            
            # Factor 3: Data Alignment Score (followed by consistent data?)
            data_alignment_score = self._calculate_data_alignment_score(row_idx, df)
            
            # Factor 4: Structural Consistency (field count vs data rows)
            structural_score = self._calculate_structural_score(row_idx, df, candidate['field_count'])
            
            # Factor 5: Junk Penalty (clear junk indicators)
            junk_penalty = self._calculate_junk_penalty(candidate['text_content'])
            
            # Combine scores
            confidence = (
                descriptor_score * 0.3 +
                data_alignment_score * 0.3 +
                structural_score * 0.2 +
                position_score * 0.1 +
                (1.0 - junk_penalty) * 0.1
            )
            
            # Build explanation
            explanation = f"desc:{descriptor_score:.2f} data:{data_alignment_score:.2f} struct:{structural_score:.2f}"
            
            header_candidate = HeaderCandidate(
                row_index=row_idx,
                confidence_score=confidence,
                field_count=candidate['field_count'],
                descriptor_score=descriptor_score,
                position_score=position_score,
                data_alignment_score=data_alignment_score,
                explanation=explanation
            )
            
            scored_candidates.append(header_candidate)
        
        return scored_candidates
    
    def _calculate_descriptor_score(self, row_data) -> float:
        """Score how much this row looks like column descriptors"""
        
        score = 0.0
        valid_fields = 0
        
        for val in row_data:
            if val is None or not str(val).strip():
                continue
                
            valid_fields += 1
            val_str = str(val).strip()
            
            # Good descriptor characteristics
            field_score = 0.0
            
            # Length - not too short, not too long
            if 2 <= len(val_str) <= 50:
                field_score += 0.3
            
            # Contains underscores or is CamelCase (common in column names)
            if '_' in val_str or any(c.isupper() for c in val_str[1:]):
                field_score += 0.2
                
            # Not primarily numeric
            if not val_str.replace('.', '').replace('-', '').replace('/', '').isdigit():
                field_score += 0.2
                
            # Common column name patterns
            column_words = ['id', 'name', 'date', 'time', 'amount', 'price', 'count', 'total', 
                           'value', 'code', 'type', 'status', 'category', 'product', 'customer',
                           'revenue', 'cost', 'profit', 'units', 'quantity', 'sales', 'margin']
            
            if any(word in val_str.lower() for word in column_words):
                field_score += 0.3
                
            score += min(field_score, 1.0)
        
        return score / max(valid_fields, 1)
    
    def _calculate_position_score(self, row_idx: int, total_rows: int) -> float:
        """Score based on position - headers usually in first 20% but not at very end"""
        
        position_ratio = row_idx / total_rows
        
        if position_ratio <= 0.1:  # Very early
            return 0.7
        elif position_ratio <= 0.2:  # Early
            return 0.9
        elif position_ratio <= 0.5:  # Middle
            return 0.6
        elif position_ratio <= 0.8:  # Later
            return 0.3
        else:  # Too late
            return 0.1
    
    def _calculate_data_alignment_score(self, row_idx: int, df: pl.DataFrame) -> float:
        """Score based on how well this aligns with data rows that follow"""
        
        score = 0.0
        
        # Look at next 3-5 rows to see if they look like data
        data_like_followers = 0
        followers_checked = 0
        
        for i in range(row_idx + 1, min(row_idx + 6, df.shape[0])):
            try:
                follower_row = df.row(i)
                followers_checked += 1
                
                # Check if follower looks like data
                non_null_count = sum(1 for val in follower_row if val is not None and str(val).strip())
                
                # Data rows should have reasonable field counts
                if non_null_count >= 2:
                    data_like_followers += 1
                    
                    # Bonus if it has mixed content (numbers + text)
                    has_numbers = any(str(val).replace('.', '').replace('-', '').isdigit() 
                                    for val in follower_row if val is not None)
                    has_text = any(not str(val).replace('.', '').replace('-', '').isdigit() 
                                 for val in follower_row if val is not None and str(val).strip())
                    
                    if has_numbers and has_text:
                        data_like_followers += 0.5
                        
            except Exception:
                continue
        
        if followers_checked > 0:
            score = data_like_followers / followers_checked
            
        return min(score, 1.0)
    
    def _calculate_structural_score(self, row_idx: int, df: pl.DataFrame, field_count: int) -> float:
        """Score based on structural consistency with nearby data"""
        
        # Compare field count with subsequent rows
        field_counts = []
        
        for i in range(row_idx + 1, min(row_idx + 10, df.shape[0])):
            try:
                row = df.row(i)
                count = sum(1 for val in row if val is not None and str(val).strip())
                field_counts.append(count)
            except Exception:
                continue
        
        if not field_counts:
            return 0.5
        
        avg_data_fields = np.mean(field_counts)
        
        # Good if header field count is similar to data field counts
        if abs(field_count - avg_data_fields) <= 2:
            return 1.0
        elif abs(field_count - avg_data_fields) <= 4:
            return 0.7
        else:
            return 0.3
    
    def _calculate_junk_penalty(self, text_content: str) -> float:
        """Penalty for clear junk patterns"""
        
        penalty = 0.0
        text_lower = text_content.lower()
        
        # Strong junk indicators
        strong_junk = ['report generated', 'total:', 'sum:', 'grand total', 'end of report', 
                       'page 1', 'printed on', 'exported on']
        
        for junk_pattern in strong_junk:
            if junk_pattern in text_lower:
                penalty += 0.8
                break  # One strong indicator is enough
        
        # Metadata patterns
        metadata_patterns = ['report', 'quarter', 'annual', 'generated', 'exported', 'created']
        metadata_count = sum(1 for pattern in metadata_patterns if pattern in text_lower)
        
        if metadata_count >= 2:  # Multiple metadata words
            penalty += 0.4
        
        return min(penalty, 1.0)
    
    def _build_result(self, df: pl.DataFrame, best_candidate: HeaderCandidate, 
                     all_candidates: List[HeaderCandidate]) -> HeaderDetectionResult:
        """Build the final header detection result"""
        
        header_row_idx = best_candidate.row_index
        
        # Get column names from the header row
        header_row = df.row(header_row_idx)
        column_names = []
        
        for i, val in enumerate(header_row):
            if val is not None and str(val).strip():
                column_names.append(str(val).strip())
            else:
                column_names.append(f"Column_{i+1}")
        
        # Identify rows to remove (everything above the header)
        rows_to_remove = list(range(header_row_idx))
        
        # Processing notes
        notes = [
            f"Analyzed {len(all_candidates)} potential header candidates",
            f"Best candidate at row {header_row_idx} with {best_candidate.field_count} fields",
            f"Will remove {len(rows_to_remove)} rows above header",
            f"Detection factors: {best_candidate.explanation}"
        ]
        
        return HeaderDetectionResult(
            header_row_index=header_row_idx,
            confidence=best_candidate.confidence_score,
            candidates=sorted(all_candidates, key=lambda x: x.confidence_score, reverse=True),
            rows_to_remove_above=rows_to_remove,
            column_names=column_names,
            processing_notes=notes
        )

def detect_and_clean_headers(df: pl.DataFrame, min_confidence: float = 0.6) -> Tuple[pl.DataFrame, HeaderDetectionResult]:
    """
    Convenience function to detect headers and return cleaned dataframe
    
    Args:
        df: Raw dataframe with unknown header position
        min_confidence: Minimum confidence threshold
        
    Returns:
        Tuple of (cleaned_df_with_proper_headers, detection_result)
    """
    
    detector = SmartHeaderDetector(min_confidence=min_confidence)
    result = detector.detect_header_row(df)
    
    # Create cleaned dataframe
    if result.header_row_index < df.shape[0]:
        # Remove rows above header
        clean_df = df[result.header_row_index:]
        
        # Set proper column names  
        if len(result.column_names) <= clean_df.shape[1]:
            # Rename columns to use detected headers
            column_mapping = {}
            for i, new_name in enumerate(result.column_names):
                if i < len(clean_df.columns):
                    column_mapping[clean_df.columns[i]] = new_name
            
            clean_df = clean_df.rename(column_mapping)
            
            # Remove the header row itself (it's now the column names)
            if clean_df.shape[0] > 1:
                clean_df = clean_df[1:]
        
        logger.info(f"Cleaned dataframe: {clean_df.shape[0]} rows x {clean_df.shape[1]} columns")
        
    else:
        logger.warning("Header detection failed, returning original dataframe")
        clean_df = df
    
    return clean_df, result