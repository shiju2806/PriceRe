"""
OpenAI-powered PriceRe Chat Engine
Provides intelligent reinsurance and data cleaning assistance using GPT-4o mini
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import polars as pl
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not installed. Install with: pip install openai")

@dataclass
class ChatContext:
    """Enhanced context for LLM chat"""
    original_df: Optional[pl.DataFrame] = None
    cleaned_df: Optional[pl.DataFrame] = None
    cleaning_summary: Optional[Dict] = None
    has_data: bool = False
    rows_original: int = 0
    rows_cleaned: int = 0
    columns: List[str] = None
    data_sample: Optional[Dict] = None

class OpenAIChatEngine:
    """OpenAI GPT-4o mini powered chat engine for PriceRe"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAI chat engine"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = "gpt-4o-mini"
        self.conversation_history = []
        self.context = ChatContext()
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize OpenAI client
        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key)
        
        logger.info("OpenAI Chat Engine initialized with GPT-4o mini")
    
    def set_data_context(self, original_df: pl.DataFrame, cleaned_df: pl.DataFrame, cleaning_result: Any = None):
        """Set data cleaning context for enhanced responses"""
        try:
            # Convert to pandas for easier processing
            original_pd = original_df.to_pandas() if original_df is not None else None
            cleaned_pd = cleaned_df.to_pandas() if cleaned_df is not None else None
            
            # Update context
            self.context = ChatContext(
                original_df=original_df,
                cleaned_df=cleaned_df,
                cleaning_summary=self._create_cleaning_summary(original_pd, cleaned_pd, cleaning_result),
                has_data=True,
                rows_original=len(original_pd) if original_pd is not None else 0,
                rows_cleaned=len(cleaned_pd) if cleaned_pd is not None else 0,
                columns=list(cleaned_pd.columns) if cleaned_pd is not None else [],
                data_sample=self._create_data_sample(cleaned_pd)
            )
            
            logger.info(f"Data context set: {self.context.rows_original} → {self.context.rows_cleaned} rows")
            
        except Exception as e:
            logger.error(f"Error setting data context: {e}")
            self.context.has_data = False
    
    def _create_cleaning_summary(self, original_df, cleaned_df, cleaning_result) -> Dict:
        """Create summary of cleaning operations"""
        if original_df is None or cleaned_df is None:
            return {}
        
        summary = {
            "rows_removed": len(original_df) - len(cleaned_df),
            "removal_percentage": ((len(original_df) - len(cleaned_df)) / len(original_df) * 100) if len(original_df) > 0 else 0,
            "columns": len(cleaned_df.columns),
            "data_types": cleaned_df.dtypes.to_dict() if hasattr(cleaned_df, 'dtypes') else {}
        }
        
        # Add cleaning result details if available
        if cleaning_result and hasattr(cleaning_result, 'junk_rows'):
            summary["junk_patterns_detected"] = len(cleaning_result.junk_rows)
        
        return summary
    
    def _create_data_sample(self, df) -> Optional[Dict]:
        """Create a sample of the data for context"""
        if df is None or len(df) == 0:
            return None
        
        try:
            # Get a small sample (first 3 rows)
            sample_data = df.head(3).to_dict('records')
            
            return {
                "sample_rows": sample_data,
                "column_info": {
                    col: {
                        "type": str(df[col].dtype),
                        "sample_values": df[col].dropna().head(3).tolist() if col in df.columns else []
                    }
                    for col in df.columns[:10]  # Limit to first 10 columns
                }
            }
        except Exception as e:
            logger.warning(f"Could not create data sample: {e}")
            return None
    
    def _build_system_prompt(self) -> str:
        """Build comprehensive system prompt for PriceRe assistant"""
        
        base_prompt = """You are PriceRe Chat, an expert AI assistant specialized in:

1. **Reinsurance & Insurance**: Treaty reinsurance, facultative reinsurance, catastrophe modeling, pricing, risk assessment, regulatory compliance, market analysis
2. **Data Cleaning & Analysis**: Identifying junk rows, data quality assessment, statistical analysis, data transformation recommendations
3. **Platform Guidance**: Help users navigate the PriceRe platform features and workflows

**Your Personality**:
- Professional but friendly and approachable
- Provide clear, actionable advice
- Ask clarifying questions when needed
- Explain technical concepts in accessible terms
- Focus on practical solutions

**Response Style**:
- Be concise but comprehensive
- Use bullet points for multiple items
- Provide specific examples when helpful
- Always consider the insurance/reinsurance context
"""

        # Add data context if available
        if self.context.has_data:
            data_context = f"""

**Current Data Context**:
- Original dataset: {self.context.rows_original:,} rows
- After cleaning: {self.context.rows_cleaned:,} rows  
- Rows removed: {self.context.rows_original - self.context.rows_cleaned:,} ({self.context.cleaning_summary.get('removal_percentage', 0):.1f}%)
- Columns: {len(self.context.columns)} ({', '.join(self.context.columns[:5])}{'...' if len(self.context.columns) > 5 else ''})

You can provide specific insights about this dataset, suggest further cleaning refinements, identify potential data quality issues, and explain what the cleaned data means for reinsurance analysis.
"""
            base_prompt += data_context
        
        base_prompt += """

**Important**: Always stay within your expertise areas. If asked about topics outside reinsurance, data cleaning, or the PriceRe platform, politely redirect to your core capabilities."""

        return base_prompt
    
    def chat(self, user_message: str) -> str:
        """Process user message and return AI response"""
        try:
            # Add user message to history
            self.conversation_history.append({
                "role": "user", 
                "content": user_message,
                "timestamp": datetime.now().isoformat()
            })
            
            # Build messages for API call
            messages = [
                {"role": "system", "content": self._build_system_prompt()}
            ]
            
            # Add conversation history (keep last 10 messages for context)
            recent_history = self.conversation_history[-10:]
            for msg in recent_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            # Extract response
            ai_response = response.choices[0].message.content.strip()
            
            # Add assistant response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": ai_response,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info(f"OpenAI response generated for: '{user_message[:50]}...'")
            return ai_response
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"I apologize, but I'm experiencing technical difficulties. Please try again. Error: {str(e)}"
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history.copy()
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def suggest_data_insights(self) -> str:
        """Generate proactive insights about the current dataset"""
        if not self.context.has_data:
            return "Upload and clean data first to get specific insights about your dataset."
        
        insight_prompt = f"""Based on the current dataset context:
- {self.context.rows_original:,} original rows → {self.context.rows_cleaned:,} cleaned rows
- {self.context.cleaning_summary.get('removal_percentage', 0):.1f}% of rows removed as junk
- Columns: {', '.join(self.context.columns[:10])}

Provide 3-4 specific insights about this data's quality and potential use in reinsurance analysis. Be practical and actionable."""

        return self.chat(insight_prompt)

# Utility functions for integration
def create_openai_engine(api_key: Optional[str] = None) -> Optional[OpenAIChatEngine]:
    """Factory function to create OpenAI engine with error handling"""
    try:
        return OpenAIChatEngine(api_key=api_key)
    except Exception as e:
        logger.error(f"Could not create OpenAI engine: {e}")
        return None

def is_openai_available() -> bool:
    """Check if OpenAI integration is available"""
    return OPENAI_AVAILABLE and bool(os.getenv("OPENAI_API_KEY"))