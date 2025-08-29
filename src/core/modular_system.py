"""
Modular System Architecture
Ties together event system, configuration, and dependency injection
Single source of truth with automatic dependency flow
"""

from typing import Dict, Any, List, Type, Optional
import asyncio
import logging
from pathlib import Path

# Import our core systems
from .event_system import event_bus, EventHandler, Event, EventType
from .configuration import config_manager, ConfigurationManager
from .dependency_injection import container, ServiceContainer
from .assumption_management import assumption_manager, AssumptionManager
from .unified_data_hub import unified_hub, UnifiedDataHub

logger = logging.getLogger(__name__)

class ModularComponent:
    """
    Base class for all system components
    Automatically integrates with event system and configuration
    """
    
    def __init__(self, name: str):
        self.name = name
        self.config = None
        self.dependencies: List[str] = []
        self.is_initialized = False
        
        # Auto-register with dependency injection
        container.register_instance(type(self), self)
    
    async def initialize(self) -> bool:
        """Initialize component with configuration and dependencies"""
        try:
            # Load configuration
            self.config = self._load_configuration()
            
            # Resolve dependencies
            await self._resolve_dependencies()
            
            # Component-specific initialization
            await self._initialize_component()
            
            # Register event handlers
            self._register_event_handlers()
            
            self.is_initialized = True
            logger.info(f"Initialized component: {self.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.name}: {e}")
            return False
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load component-specific configuration"""
        # Override in subclasses
        return {}
    
    async def _resolve_dependencies(self):
        """Resolve component dependencies"""
        # Override in subclasses if needed
        pass
    
    async def _initialize_component(self):
        """Component-specific initialization logic"""
        # Override in subclasses
        pass
    
    def _register_event_handlers(self):
        """Register event handlers with event bus"""
        # Override in subclasses
        pass
    
    async def dispose(self):
        """Clean shutdown of component"""
        self.is_initialized = False
        logger.info(f"Disposed component: {self.name}")

class DataSourceComponent(ModularComponent, EventHandler):
    """
    Base class for all data source components
    Automatically handles configuration, caching, and event emission
    """
    
    def __init__(self, name: str):
        ModularComponent.__init__(self, name)
        EventHandler.__init__(self, name)
        
        self.cache_duration_hours = 24
        self.last_refresh = None
        self.cached_data = None
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load data source configuration"""
        ds_config = config_manager.get_data_source_config(self.name)
        if ds_config:
            return ds_config.__dict__
        return {}
    
    async def get_data(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Get data with automatic caching and event emission"""
        
        # Check cache first
        if not force_refresh and self._is_cache_valid():
            return self.cached_data
        
        try:
            # Fetch fresh data
            data = await self._fetch_data()
            
            # Update cache
            self.cached_data = data
            self.last_refresh = datetime.now()
            
            # Emit event about data update
            await event_bus.emit(Event(
                event_type=EventType.ECONOMIC_DATA_UPDATED if 'economic' in self.name 
                          else EventType.MORTALITY_DATA_UPDATED,
                source=self.name,
                timestamp=datetime.now(),
                data={
                    'source': self.name,
                    'data_keys': list(data.keys()) if isinstance(data, dict) else [],
                    'record_count': len(data) if isinstance(data, (list, dict)) else 1
                }
            ))
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data from {self.name}: {e}")
            
            # Return cached data if available
            if self.cached_data:
                logger.warning(f"Using cached data for {self.name}")
                return self.cached_data
            
            raise
    
    def _is_cache_valid(self) -> bool:
        """Check if cached data is still valid"""
        if not self.cached_data or not self.last_refresh:
            return False
        
        from datetime import datetime, timedelta
        cache_expiry = self.last_refresh + timedelta(hours=self.cache_duration_hours)
        return datetime.now() < cache_expiry
    
    async def _fetch_data(self) -> Dict[str, Any]:
        """Fetch fresh data - override in subclasses"""
        raise NotImplementedError
    
    async def handle_event(self, event: Event) -> Optional[List[Event]]:
        """Handle events that might require data refresh"""
        if event.event_type == EventType.ASSUMPTION_CHANGED:
            # Assumption changed, might need to refresh data
            await self.get_data(force_refresh=True)
        
        return None

class ModelComponent(ModularComponent, EventHandler):
    """
    Base class for all ML model components
    Handles training, validation, and performance tracking
    """
    
    def __init__(self, name: str):
        ModularComponent.__init__(self, name)
        EventHandler.__init__(self, name)
        
        self.model = None
        self.last_trained = None
        self.performance_metrics = {}
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load model configuration"""
        model_config = config_manager.get_model_config(self.name)
        if model_config:
            return model_config.__dict__
        return {}
    
    async def _initialize_component(self):
        """Initialize model component"""
        # Load existing model or create new one
        await self._load_or_create_model()
        
        # Schedule periodic retraining if configured
        if self.config.get('auto_retrain', False):
            self._schedule_retraining()
    
    async def _load_or_create_model(self):
        """Load existing model or create new one"""
        # Override in subclasses
        pass
    
    def _schedule_retraining(self):
        """Schedule automatic retraining"""
        # Override in subclasses to implement scheduling
        pass
    
    async def train(self, training_data: Any) -> Dict[str, float]:
        """Train the model and return performance metrics"""
        try:
            # Train model
            metrics = await self._train_model(training_data)
            
            # Update performance tracking
            self.performance_metrics = metrics
            self.last_trained = datetime.now()
            
            # Emit model training event
            await event_bus.emit(Event(
                event_type=EventType.MODEL_RETRAINED,
                source=self.name,
                timestamp=datetime.now(),
                data={
                    'model_name': self.name,
                    'performance_metrics': metrics,
                    'training_size': len(training_data) if hasattr(training_data, '__len__') else 'unknown'
                }
            ))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model {self.name}: {e}")
            raise
    
    async def _train_model(self, training_data: Any) -> Dict[str, float]:
        """Train the actual model - override in subclasses"""
        raise NotImplementedError
    
    async def predict(self, input_data: Any) -> Any:
        """Make prediction using the model"""
        if not self.model:
            raise ValueError(f"Model {self.name} not trained")
        
        return await self._make_prediction(input_data)
    
    async def _make_prediction(self, input_data: Any) -> Any:
        """Make actual prediction - override in subclasses"""
        raise NotImplementedError
    
    async def handle_event(self, event: Event) -> Optional[List[Event]]:
        """Handle events that might trigger model updates"""
        if event.event_type == EventType.PRICING_SESSION_COMPLETED:
            # New pricing data available, consider retraining
            if self._should_retrain():
                # Schedule retraining
                return [Event(
                    event_type=EventType.MODEL_RETRAINED,
                    source=self.name,
                    timestamp=datetime.now(),
                    data={'trigger': 'new_pricing_data'}
                )]
        
        return None
    
    def _should_retrain(self) -> bool:
        """Determine if model should be retrained"""
        # Override in subclasses with specific logic
        return False

class SystemOrchestrator:
    """
    Central orchestrator that manages all system components
    Ensures proper initialization order and dependency flow
    """
    
    def __init__(self):
        self.components: Dict[str, ModularComponent] = {}
        self.initialization_order: List[str] = []
        self.is_running = False
        
        logger.info("SystemOrchestrator initialized")
    
    def register_component(self, component: ModularComponent, 
                          dependencies: Optional[List[str]] = None):
        """Register a component with the system"""
        self.components[component.name] = component
        
        if dependencies:
            component.dependencies = dependencies
        
        logger.info(f"Registered component: {component.name}")
    
    async def initialize_system(self) -> bool:
        """Initialize all components in dependency order"""
        try:
            # Calculate initialization order
            self.initialization_order = self._calculate_init_order()
            
            # Initialize configuration system first
            config_issues = config_manager.validate_configuration()
            if config_issues['errors']:
                logger.error(f"Configuration errors: {config_issues['errors']}")
                return False
            
            # Initialize components in order
            for component_name in self.initialization_order:
                component = self.components[component_name]
                success = await component.initialize()
                
                if not success:
                    logger.error(f"Failed to initialize {component_name}")
                    return False
            
            # Start event processing
            await event_bus.start_processing()
            
            self.is_running = True
            logger.info("System initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    def _calculate_init_order(self) -> List[str]:
        """Calculate component initialization order based on dependencies"""
        # Simple topological sort
        order = []
        visited = set()
        temp_visited = set()
        
        def visit(component_name: str):
            if component_name in temp_visited:
                raise ValueError(f"Circular dependency involving {component_name}")
            
            if component_name in visited:
                return
            
            temp_visited.add(component_name)
            
            # Visit dependencies first
            component = self.components[component_name]
            for dep in component.dependencies:
                if dep in self.components:
                    visit(dep)
            
            temp_visited.remove(component_name)
            visited.add(component_name)
            order.append(component_name)
        
        for component_name in self.components:
            if component_name not in visited:
                visit(component_name)
        
        return order
    
    async def shutdown_system(self):
        """Shutdown all components gracefully"""
        if not self.is_running:
            return
        
        logger.info("Starting system shutdown")
        
        # Stop event processing
        await event_bus.stop_processing()
        
        # Dispose components in reverse order
        for component_name in reversed(self.initialization_order):
            component = self.components[component_name]
            try:
                await component.dispose()
            except Exception as e:
                logger.warning(f"Error disposing {component_name}: {e}")
        
        # Dispose service container
        container.dispose()
        
        self.is_running = False
        logger.info("System shutdown complete")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'is_running': self.is_running,
            'components': {
                name: comp.is_initialized 
                for name, comp in self.components.items()
            },
            'initialization_order': self.initialization_order,
            'event_system': event_bus.get_system_health(),
            'configuration': config_manager.get_configuration_summary(),
            'dependencies': container.get_dependency_graph()
        }

# Global system orchestrator
system_orchestrator = SystemOrchestrator()

# Register core components
system_orchestrator.register_component(ModularComponent("config_manager"))
system_orchestrator.register_component(ModularComponent("event_bus"))
system_orchestrator.register_component(ModularComponent("dependency_container"))
system_orchestrator.register_component(ModularComponent("assumption_manager"), ["config_manager"])
system_orchestrator.register_component(ModularComponent("unified_hub"), ["assumption_manager", "event_bus"])