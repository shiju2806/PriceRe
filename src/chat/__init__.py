"""
PriceRe Chat Assistant Module
Provides natural language interface for data cleaning refinement
"""

from .chat_assistant import PriceReChatAssistant, ChatIntent, ChatAction
from .streamlit_chat_interface import (
    render_chat_interface, 
    render_chat_sidebar,
    show_chat_in_expander,
    add_chat_refinement_button,
    reset_chat_session
)

__version__ = "1.0.0"
__all__ = [
    'PriceReChatAssistant',
    'ChatIntent', 
    'ChatAction',
    'render_chat_interface',
    'render_chat_sidebar', 
    'show_chat_in_expander',
    'add_chat_refinement_button',
    'reset_chat_session'
]