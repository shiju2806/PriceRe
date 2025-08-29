"""
Event-Driven Architecture System
Central event bus for automatic dependency updates
"""

from typing import Dict, List, Any, Callable, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging
import asyncio
from weakref import WeakSet
import json

logger = logging.getLogger(__name__)

class EventType(Enum):
    # Data source events
    ECONOMIC_DATA_UPDATED = "economic_data_updated"
    MORTALITY_DATA_UPDATED = "mortality_data_updated"
    ASSUMPTION_CHANGED = "assumption_changed"
    
    # Pricing events
    PRICING_SESSION_CREATED = "pricing_session_created"
    PRICING_SESSION_COMPLETED = "pricing_session_completed"
    PRICING_RESULTS_AVAILABLE = "pricing_results_available"
    
    # Model events
    MODEL_PERFORMANCE_UPDATED = "model_performance_updated"
    MODEL_RETRAINED = "model_retrained"
    MODEL_VALIDATION_COMPLETED = "model_validation_completed"
    
    # Business events
    ASSUMPTION_OVERRIDE_REQUESTED = "assumption_override_requested"
    ASSUMPTION_OVERRIDE_APPROVED = "assumption_override_approved"
    REGULATORY_FILING_REQUIRED = "regulatory_filing_required"
    
    # UI events
    DASHBOARD_REFRESH_REQUESTED = "dashboard_refresh_requested"
    USER_SESSION_CREATED = "user_session_created"

@dataclass
class Event:
    """Single event in the system"""
    event_type: EventType
    source: str
    timestamp: datetime
    data: Dict[str, Any]
    correlation_id: Optional[str] = None  # For tracing event chains

class EventHandler:
    """Base class for event handlers"""
    
    def __init__(self, name: str):
        self.name = name
        self.handled_events: List[EventType] = []
    
    async def handle_event(self, event: Event) -> Optional[List[Event]]:
        """
        Handle an event and optionally return new events to emit
        Returns list of new events or None
        """
        raise NotImplementedError
    
    def can_handle(self, event_type: EventType) -> bool:
        """Check if this handler can process this event type"""
        return event_type in self.handled_events

class EventBus:
    """
    Central event bus - single source of truth for all system events
    Automatically propagates changes throughout the system
    """
    
    def __init__(self):
        self.handlers: Dict[EventType, WeakSet] = {}
        self.event_history: List[Event] = []
        self.max_history = 10000  # Keep last 10k events
        
        # Event processing queue
        self.event_queue = asyncio.Queue()
        self.processing_task = None
        self.is_running = False
        
        logger.info("EventBus initialized")
    
    def register_handler(self, handler: EventHandler, event_types: List[EventType]):
        """Register handler for specific event types"""
        handler.handled_events = event_types
        
        for event_type in event_types:
            if event_type not in self.handlers:
                self.handlers[event_type] = WeakSet()
            self.handlers[event_type].add(handler)
        
        logger.info(f"Registered handler {handler.name} for {len(event_types)} event types")
    
    async def emit(self, event: Event):
        """Emit event to all registered handlers"""
        logger.debug(f"Emitting event: {event.event_type.value} from {event.source}")
        
        # Store in history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
        
        # Queue for processing
        await self.event_queue.put(event)
    
    async def start_processing(self):
        """Start event processing loop"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_events())
        logger.info("Event processing started")
    
    async def stop_processing(self):
        """Stop event processing"""
        self.is_running = False
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        logger.info("Event processing stopped")
    
    async def _process_events(self):
        """Main event processing loop"""
        while self.is_running:
            try:
                event = await self.event_queue.get()
                await self._handle_single_event(event)
                self.event_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _handle_single_event(self, event: Event):
        """Handle a single event by dispatching to all relevant handlers"""
        if event.event_type not in self.handlers:
            logger.debug(f"No handlers for event type: {event.event_type.value}")
            return
        
        # Get all handlers for this event type
        handlers = list(self.handlers[event.event_type])  # Convert WeakSet to list
        
        # Process handlers concurrently
        tasks = []
        for handler in handlers:
            if handler.can_handle(event.event_type):
                tasks.append(self._safe_handle_event(handler, event))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Emit any new events returned by handlers
            for result in results:
                if isinstance(result, list):  # Handler returned new events
                    for new_event in result:
                        if isinstance(new_event, Event):
                            await self.event_queue.put(new_event)
    
    async def _safe_handle_event(self, handler: EventHandler, event: Event) -> Optional[List[Event]]:
        """Safely handle event with error catching"""
        try:
            return await handler.handle_event(event)
        except Exception as e:
            logger.error(f"Handler {handler.name} failed for event {event.event_type.value}: {e}")
            return None
    
    def get_event_history(self, event_types: Optional[List[EventType]] = None, 
                         hours_back: Optional[int] = None) -> List[Event]:
        """Get event history with optional filtering"""
        events = self.event_history
        
        if hours_back:
            cutoff = datetime.now() - datetime.timedelta(hours=hours_back)
            events = [e for e in events if e.timestamp > cutoff]
        
        if event_types:
            events = [e for e in events if e.event_type in event_types]
        
        return events
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get event system health metrics"""
        return {
            "is_running": self.is_running,
            "queue_size": self.event_queue.qsize() if hasattr(self.event_queue, 'qsize') else 0,
            "total_handlers": sum(len(handlers) for handlers in self.handlers.values()),
            "event_types_registered": len(self.handlers),
            "events_processed_today": len([e for e in self.event_history 
                                         if e.timestamp.date() == datetime.now().date()]),
            "last_event": self.event_history[-1].event_type.value if self.event_history else None
        }

# Global event bus instance - single source of truth for all events
event_bus = EventBus()