"""
PriceRe Chat Assistant for Data Cleaning Refinement
Supplements automated cleaning with natural language interaction
Now with OpenAI GPT-4o mini integration for intelligent conversations
"""

import logging
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import polars as pl
import numpy as np
from datetime import datetime

# Try to import OpenAI engine
try:
    from .openai_chat_engine import OpenAIChatEngine, create_openai_engine, is_openai_available
    OPENAI_ENGINE_AVAILABLE = True
except ImportError:
    OPENAI_ENGINE_AVAILABLE = False

logger = logging.getLogger(__name__)

class ChatIntent(Enum):
    """User intent classification"""
    REMOVE_ROWS = "remove_rows"
    RESTORE_ROWS = "restore_rows" 
    FIND_PATTERN = "find_pattern"
    ASK_QUESTION = "ask_question"
    SHOW_EXAMPLES = "show_examples"
    CONFIRM_ACTION = "confirm_action"
    UNDO_ACTION = "undo_action"
    UNKNOWN = "unknown"

@dataclass
class ChatContext:
    """Context information for chat assistant"""
    original_df: pl.DataFrame
    current_cleaned_df: pl.DataFrame
    original_cleaning_result: Any  # HybridResult from Phase 2
    removed_row_indices: List[int]
    session_history: List[Dict]
    domain: str = "insurance"
    
@dataclass 
class ChatMessage:
    """Chat message structure"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    metadata: Optional[Dict] = None

@dataclass
class ChatAction:
    """Action to be taken based on chat"""
    intent: ChatIntent
    description: str
    row_indices: List[int]
    confidence: float
    preview_data: Optional[List[Dict]] = None

class DataPatternAnalyzer:
    """Analyzes data patterns for chat understanding"""
    
    def __init__(self):
        self.insurance_terms = [
            'policy', 'premium', 'claim', 'coverage', 'deductible',
            'liability', 'retention', 'reinsurance', 'cedant', 'treaty',
            'facultative', 'quota share', 'surplus', 'excess',
            'attachment point', 'aggregate', 'occurrence'
        ]
    
    def find_rows_by_description(self, df: pl.DataFrame, description: str) -> List[int]:
        """Find rows matching natural language description"""
        description_lower = description.lower()
        matching_rows = []
        
        # Convert to list of dicts for easier processing
        rows_data = df.to_dicts()
        
        for idx, row_dict in enumerate(rows_data):
            row_text = ' '.join(str(v) for v in row_dict.values() if v is not None).lower()
            
            # Pattern matching based on description
            if self._matches_description(row_text, description_lower):
                matching_rows.append(idx)
        
        return matching_rows
    
    def _matches_description(self, row_text: str, description: str) -> bool:
        """Check if row matches description"""
        # Summary/total patterns
        if any(word in description for word in ['summary', 'total', 'sum']):
            if any(word in row_text for word in ['total', 'sum', 'summary', 'subtotal', 'grand']):
                return True
        
        # Header patterns  
        if 'header' in description:
            # Look for column-like names
            if any(term in row_text for term in self.insurance_terms):
                return True
        
        # Position-based patterns
        if 'bottom' in description or 'footer' in description:
            # This would need position context from caller
            return any(word in row_text for word in ['generated', 'report', 'end', 'footer'])
        
        if 'top' in description:
            return any(word in row_text for word in ['report', 'title', 'generated'])
        
        # Content-based patterns
        if 'empty' in description:
            return len(row_text.strip()) < 10
        
        return False
    
    def analyze_removed_rows(self, df: pl.DataFrame, removed_indices: List[int]) -> Dict[str, List[int]]:
        """Categorize removed rows for better understanding"""
        categories = {
            'headers': [],
            'summaries': [], 
            'metadata': [],
            'empty': [],
            'unknown': []
        }
        
        rows_data = df.to_dicts()
        
        for idx in removed_indices:
            if idx < len(rows_data):
                row_dict = rows_data[idx]
                row_text = ' '.join(str(v) for v in row_dict.values() if v is not None).lower()
                
                if self._is_header_like(row_text):
                    categories['headers'].append(idx)
                elif self._is_summary_like(row_text):
                    categories['summaries'].append(idx)
                elif self._is_metadata_like(row_text):
                    categories['metadata'].append(idx)
                elif len(row_text.strip()) < 5:
                    categories['empty'].append(idx)
                else:
                    categories['unknown'].append(idx)
        
        return categories
    
    def _is_header_like(self, text: str) -> bool:
        """Check if text looks like headers"""
        return any(term in text for term in self.insurance_terms)
    
    def _is_summary_like(self, text: str) -> bool:
        """Check if text looks like summaries"""
        return any(word in text for word in ['total', 'sum', 'summary', 'count', 'subtotal'])
    
    def _is_metadata_like(self, text: str) -> bool:
        """Check if text looks like metadata"""
        return any(word in text for word in ['generated', 'report', 'exported', 'created', 'system'])

class NaturalLanguageProcessor:
    """Process natural language queries for data cleaning"""
    
    def __init__(self):
        self.pattern_analyzer = DataPatternAnalyzer()
    
    def classify_intent(self, user_message: str) -> ChatIntent:
        """Classify user intent from message"""
        message_lower = user_message.lower()
        
        # Remove/delete intent
        if any(word in message_lower for word in ['remove', 'delete', 'get rid of', 'eliminate']):
            return ChatIntent.REMOVE_ROWS
        
        # Restore/add back intent
        if any(word in message_lower for word in ['restore', 'add back', 'bring back', 'keep']):
            return ChatIntent.RESTORE_ROWS
        
        # Find/show intent
        if any(word in message_lower for word in ['find', 'show', 'where', 'which']):
            return ChatIntent.FIND_PATTERN
        
        # Question intent
        if any(word in message_lower for word in ['what', 'why', 'how', 'explain']):
            return ChatIntent.ASK_QUESTION
        
        # Confirmation intent
        if any(word in message_lower for word in ['yes', 'correct', 'right', 'ok', 'sure']):
            return ChatIntent.CONFIRM_ACTION
        
        # Undo intent
        if any(word in message_lower for word in ['undo', 'revert', 'go back']):
            return ChatIntent.UNDO_ACTION
        
        return ChatIntent.UNKNOWN
    
    def extract_action(self, user_message: str, context: ChatContext) -> ChatAction:
        """Extract actionable information from user message"""
        intent = self.classify_intent(user_message)
        
        if intent == ChatIntent.REMOVE_ROWS:
            return self._extract_remove_action(user_message, context)
        elif intent == ChatIntent.RESTORE_ROWS:
            return self._extract_restore_action(user_message, context)
        elif intent == ChatIntent.FIND_PATTERN:
            return self._extract_find_action(user_message, context)
        else:
            return ChatAction(
                intent=intent,
                description=user_message,
                row_indices=[],
                confidence=0.5
            )
    
    def _extract_remove_action(self, message: str, context: ChatContext) -> ChatAction:
        """Extract rows to remove from message"""
        # Find rows matching the description
        target_rows = self.pattern_analyzer.find_rows_by_description(
            context.current_cleaned_df, message
        )
        
        # Create preview data
        preview_data = []
        if target_rows:
            df_dicts = context.current_cleaned_df.to_dicts()
            preview_data = [
                {"row_index": idx, "data": df_dicts[idx]} 
                for idx in target_rows[:5]  # Show first 5 matches
            ]
        
        return ChatAction(
            intent=ChatIntent.REMOVE_ROWS,
            description=f"Remove {len(target_rows)} rows matching: {message}",
            row_indices=target_rows,
            confidence=0.8 if target_rows else 0.2,
            preview_data=preview_data
        )
    
    def _extract_restore_action(self, message: str, context: ChatContext) -> ChatAction:
        """Extract rows to restore from message"""
        # Find rows in the originally removed set that match description
        removed_df = context.original_df.filter(
            pl.int_range(pl.len()).is_in(context.removed_row_indices)
        )
        
        target_rows = self.pattern_analyzer.find_rows_by_description(removed_df, message)
        # Convert back to original indices
        original_target_indices = [context.removed_row_indices[idx] for idx in target_rows]
        
        # Create preview data
        preview_data = []
        if target_rows:
            removed_dicts = removed_df.to_dicts()
            preview_data = [
                {"row_index": context.removed_row_indices[idx], "data": removed_dicts[idx]}
                for idx in target_rows[:5]
            ]
        
        return ChatAction(
            intent=ChatIntent.RESTORE_ROWS,
            description=f"Restore {len(target_rows)} rows matching: {message}",
            row_indices=original_target_indices,
            confidence=0.8 if target_rows else 0.2,
            preview_data=preview_data
        )
    
    def _extract_find_action(self, message: str, context: ChatContext) -> ChatAction:
        """Extract pattern finding request"""
        target_rows = self.pattern_analyzer.find_rows_by_description(
            context.current_cleaned_df, message
        )
        
        return ChatAction(
            intent=ChatIntent.FIND_PATTERN,
            description=f"Found {len(target_rows)} rows matching: {message}",
            row_indices=target_rows,
            confidence=0.7,
            preview_data=[
                {"row_index": idx, "data": context.current_cleaned_df.to_dicts()[idx]}
                for idx in target_rows[:10]
            ]
        )

class PriceReChatAssistant:
    """Main chat assistant for PriceRe with OpenAI integration"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.nlp = NaturalLanguageProcessor()
        self.conversation_history: List[ChatMessage] = []
        self.action_history: List[ChatAction] = []
        self.context: Optional[ChatContext] = None
        
        # Initialize OpenAI engine if available
        self.openai_engine = None
        self.use_openai = False
        
        if OPENAI_ENGINE_AVAILABLE:
            try:
                self.openai_engine = create_openai_engine(openai_api_key)
                self.use_openai = self.openai_engine is not None
                logger.info(f"OpenAI integration: {'✅ Enabled' if self.use_openai else '❌ Disabled'}")
            except Exception as e:
                logger.warning(f"OpenAI initialization failed: {e}")
                self.use_openai = False
        else:
            logger.info("OpenAI integration not available - using basic pattern matching")
    
    def initialize_context(self, original_df: pl.DataFrame, cleaned_df: pl.DataFrame, 
                          cleaning_result: Any) -> None:
        """Initialize chat context with cleaning results"""
        self.context = ChatContext(
            original_df=original_df,
            current_cleaned_df=cleaned_df,
            original_cleaning_result=cleaning_result,
            removed_row_indices=getattr(cleaning_result, 'junk_row_indices', []),
            session_history=[]
        )
        
        # Initialize OpenAI engine with data context if available
        if self.use_openai and self.openai_engine:
            try:
                self.openai_engine.set_data_context(original_df, cleaned_df, cleaning_result)
                logger.info("OpenAI engine updated with data context")
            except Exception as e:
                logger.warning(f"Failed to set OpenAI data context: {e}")
        
        # Add initial context message
        initial_message = self._generate_initial_message()
        self.add_message("assistant", initial_message)
    
    def process_user_message(self, user_message: str) -> Tuple[str, Optional[ChatAction]]:
        """Process user message and return response with optional action"""
        # Add user message to history
        self.add_message("user", user_message)
        
        # Use OpenAI if available (context not required for general questions)
        if self.use_openai and self.openai_engine:
            try:
                response = self.openai_engine.chat(user_message)
                self.add_message("assistant", response)
                
                # For OpenAI responses, we can still try to extract actions for data manipulation
                action = None
                if self.context:
                    action = self.nlp.extract_action(user_message, self.context)
                    if action.confidence <= 0.5:
                        action = None
                
                return response, action
                
            except Exception as e:
                logger.error(f"OpenAI processing failed: {e}")
                # Fall back to basic processing
        
        # Basic pattern matching fallback
        if not self.context:
            response = "I can help with reinsurance questions, platform guidance, and data analysis. Upload and clean data for enhanced features!"
            self.add_message("assistant", response)
            return response, None
        
        # Extract action from message using basic NLP
        action = self.nlp.extract_action(user_message, self.context)
        
        # Generate response based on action
        if action.intent == ChatIntent.REMOVE_ROWS:
            response = self._handle_remove_request(action)
        elif action.intent == ChatIntent.RESTORE_ROWS:
            response = self._handle_restore_request(action)
        elif action.intent == ChatIntent.FIND_PATTERN:
            response = self._handle_find_request(action)
        elif action.intent == ChatIntent.ASK_QUESTION:
            response = self._handle_question(user_message)
        else:
            response = "I can help with data cleaning refinements, reinsurance questions, and platform guidance. What would you like to know?"
        
        # Add assistant response to history
        self.add_message("assistant", response)
        
        return response, action if action.confidence > 0.5 else None
    
    def apply_action(self, action: ChatAction) -> Tuple[pl.DataFrame, str]:
        """Apply the chat action to the data"""
        if not self.context:
            return self.context.current_cleaned_df, "No context available"
        
        if action.intent == ChatIntent.REMOVE_ROWS:
            # Remove specified rows from current cleaned data
            mask = pl.int_range(pl.len()).is_in(action.row_indices).not_()
            new_df = self.context.current_cleaned_df.filter(mask)
            
            self.context.current_cleaned_df = new_df
            self.action_history.append(action)
            
            return new_df, f"✅ Removed {len(action.row_indices)} rows"
        
        elif action.intent == ChatIntent.RESTORE_ROWS:
            # Restore rows from original data
            restore_df = self.context.original_df.filter(
                pl.int_range(pl.len()).is_in(action.row_indices)
            )
            
            # Combine with current cleaned data (this is simplified - would need proper merging)
            new_df = pl.concat([self.context.current_cleaned_df, restore_df])
            
            self.context.current_cleaned_df = new_df
            self.action_history.append(action)
            
            return new_df, f"✅ Restored {len(action.row_indices)} rows"
        
        return self.context.current_cleaned_df, "No action taken"
    
    def add_message(self, role: str, content: str) -> None:
        """Add message to conversation history"""
        self.conversation_history.append(
            ChatMessage(
                role=role,
                content=content, 
                timestamp=datetime.now()
            )
        )
    
    def get_conversation_history(self) -> List[ChatMessage]:
        """Get full conversation history"""
        return self.conversation_history
    
    def _generate_initial_message(self) -> str:
        """Generate initial context message"""
        if not self.context:
            return "Chat assistant ready"
        
        original_rows = len(self.context.original_df)
        cleaned_rows = len(self.context.current_cleaned_df) 
        removed_count = len(self.context.removed_row_indices)
        
        # Analyze what was removed
        removed_analysis = self.nlp.pattern_analyzer.analyze_removed_rows(
            self.context.original_df, self.context.removed_row_indices
        )
        
        analysis_text = []
        for category, indices in removed_analysis.items():
            if indices:
                analysis_text.append(f"{len(indices)} {category}")
        
        message = f"""🤖 **PriceRe Cleaning Assistant Ready**

📊 **Auto-cleaning completed:**
- Original rows: {original_rows:,}
- Clean rows: {cleaned_rows:,} 
- Removed: {removed_count:,} rows ({removed_analysis})

💬 **What I can help with:**
- "Remove all summary rows at the bottom"
- "Restore rows with policy data"
- "Show me what you removed"
- "Find rows containing premium totals"

What would you like to refine?"""
        
        return message
    
    def _handle_remove_request(self, action: ChatAction) -> str:
        """Handle remove rows request"""
        if not action.row_indices:
            return f"❌ I couldn't find any rows matching that description. Could you be more specific?"
        
        preview_text = ""
        if action.preview_data:
            preview_text = "\n\n📋 **Preview of rows to remove:**\n"
            for item in action.preview_data[:3]:
                row_data = item['data']
                sample_values = [str(v) for v in list(row_data.values())[:3] if v is not None]
                preview_text += f"- Row {item['row_index']}: {', '.join(sample_values)}...\n"
        
        return f"""🎯 **Found {len(action.row_indices)} rows to remove**
{action.description}
{preview_text}
Type "yes" to confirm removal, or describe what you'd like to adjust."""
    
    def _handle_restore_request(self, action: ChatAction) -> str:
        """Handle restore rows request"""
        if not action.row_indices:
            return f"❌ I couldn't find any removed rows matching that description."
        
        return f"""🔄 **Found {len(action.row_indices)} rows to restore**
{action.description}

These rows were previously removed by auto-cleaning. Type "yes" to restore them."""
    
    def _handle_find_request(self, action: ChatAction) -> str:
        """Handle find pattern request"""
        if not action.row_indices:
            return f"❌ No rows found matching that pattern."
        
        preview_text = ""
        if action.preview_data:
            preview_text = "\n\n📋 **Matching rows:**\n"
            for item in action.preview_data[:5]:
                row_data = item['data']
                sample_values = [str(v) for v in list(row_data.values())[:4] if v is not None]
                preview_text += f"- Row {item['row_index']}: {', '.join(sample_values)}...\n"
        
        return f"""🔍 **Found {len(action.row_indices)} matching rows**
{preview_text}
Would you like me to remove these, or do something else with them?"""
    
    def _handle_question(self, question: str) -> str:
        """Handle general questions"""
        question_lower = question.lower()
        
        if 'why' in question_lower and ('remove' in question_lower or 'delete' in question_lower):
            return """🤔 **Why rows were removed:**
The auto-cleaning system identified these patterns as likely junk:
- Rows with summary/total information
- Metadata like "Report Generated: ..."
- Empty or mostly empty rows
- Header information above the real data

I can show you exactly what was removed - just ask "show me what was removed"."""
        
        elif 'how' in question_lower:
            return """🔧 **How the cleaning works:**
1. **Statistical Analysis**: Identifies rows with unusual content patterns
2. **Semantic Analysis**: Uses AI to understand row content similarity
3. **Header Detection**: Finds the real column headers
4. **Pattern Recognition**: Learns from data structure

You can refine any of these decisions through our chat!"""
        
        else:
            return """❓ **I can answer questions about:**
- Why specific rows were removed
- How the cleaning algorithm works  
- What patterns were detected
- Insurance/reinsurance data terminology

What would you like to know?"""