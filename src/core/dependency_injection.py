"""
Dependency Injection System
Modular component system where dependencies are injected automatically
No hard-coded imports or tight coupling
"""

from typing import Dict, Any, Type, Optional, List, Callable, TypeVar, Generic
import inspect
from dataclasses import dataclass
import logging
from weakref import WeakKeyDictionary

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class ServiceRegistration:
    """Registration information for a service"""
    service_type: Type
    implementation: Type
    singleton: bool = True
    factory: Optional[Callable] = None
    dependencies: List[str] = None
    
class ServiceContainer:
    """
    Dependency injection container
    Manages all service dependencies and automatic injection
    """
    
    def __init__(self):
        self.registrations: Dict[str, ServiceRegistration] = {}
        self.instances: Dict[str, Any] = {}  # Singleton instances
        self.dependency_graph: Dict[str, List[str]] = {}
        self.resolution_stack: List[str] = []  # For circular dependency detection
        
        logger.info("ServiceContainer initialized")
    
    def register(self, 
                interface: Type[T], 
                implementation: Type[T], 
                singleton: bool = True,
                factory: Optional[Callable] = None) -> 'ServiceContainer':
        """Register a service implementation"""
        
        service_name = interface.__name__
        
        # Analyze dependencies from constructor
        dependencies = self._analyze_dependencies(implementation)
        
        registration = ServiceRegistration(
            service_type=interface,
            implementation=implementation,
            singleton=singleton,
            factory=factory,
            dependencies=dependencies
        )
        
        self.registrations[service_name] = registration
        self.dependency_graph[service_name] = dependencies
        
        logger.info(f"Registered service: {service_name} -> {implementation.__name__}")
        return self
    
    def register_instance(self, interface: Type[T], instance: T) -> 'ServiceContainer':
        """Register a pre-created instance"""
        service_name = interface.__name__
        self.instances[service_name] = instance
        
        logger.info(f"Registered instance: {service_name}")
        return self
    
    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> 'ServiceContainer':
        """Register a factory function"""
        return self.register(interface, None, singleton=False, factory=factory)
    
    def resolve(self, interface: Type[T]) -> T:
        """Resolve a service and all its dependencies"""
        service_name = interface.__name__
        
        # Check if already have singleton instance
        if service_name in self.instances:
            return self.instances[service_name]
        
        # Check if service is registered
        if service_name not in self.registrations:
            raise ValueError(f"Service {service_name} is not registered")
        
        return self._resolve_service(service_name)
    
    def _resolve_service(self, service_name: str) -> Any:
        """Internal service resolution with circular dependency detection"""
        
        # Circular dependency check
        if service_name in self.resolution_stack:
            cycle = self.resolution_stack[self.resolution_stack.index(service_name):] + [service_name]
            raise ValueError(f"Circular dependency detected: {' -> '.join(cycle)}")
        
        # Already resolved singleton
        if service_name in self.instances:
            return self.instances[service_name]
        
        registration = self.registrations[service_name]
        
        # Add to resolution stack
        self.resolution_stack.append(service_name)
        
        try:
            # Create instance
            if registration.factory:
                instance = registration.factory()
            else:
                # Resolve dependencies first
                dependency_instances = {}
                for dep_name in registration.dependencies or []:
                    dependency_instances[dep_name] = self._resolve_service(dep_name)
                
                # Create instance with dependencies
                instance = self._create_instance(registration, dependency_instances)
            
            # Store singleton
            if registration.singleton:
                self.instances[service_name] = instance
            
            logger.debug(f"Resolved service: {service_name}")
            return instance
            
        finally:
            # Remove from resolution stack
            self.resolution_stack.pop()
    
    def _create_instance(self, registration: ServiceRegistration, dependencies: Dict[str, Any]) -> Any:
        """Create instance with dependency injection"""
        implementation = registration.implementation
        
        # Get constructor signature
        sig = inspect.signature(implementation.__init__)
        constructor_args = {}
        
        # Map dependencies to constructor parameters
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            # Try to find dependency by parameter name
            if param_name in dependencies:
                constructor_args[param_name] = dependencies[param_name]
            else:
                # Try to find by parameter type annotation
                param_type = param.annotation
                if param_type != inspect.Parameter.empty:
                    type_name = param_type.__name__ if hasattr(param_type, '__name__') else str(param_type)
                    if type_name in dependencies:
                        constructor_args[type_name] = dependencies[type_name]
        
        # Create instance
        return implementation(**constructor_args)
    
    def _analyze_dependencies(self, implementation: Type) -> List[str]:
        """Analyze constructor dependencies"""
        if implementation is None:
            return []
        
        sig = inspect.signature(implementation.__init__)
        dependencies = []
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            # Use type annotation to determine dependency
            if param.annotation != inspect.Parameter.empty:
                dep_type = param.annotation
                if hasattr(dep_type, '__name__'):
                    dependencies.append(dep_type.__name__)
                else:
                    # Handle generic types, etc.
                    dependencies.append(str(dep_type))
        
        return dependencies
    
    def validate_dependencies(self) -> Dict[str, List[str]]:
        """Validate all dependencies can be resolved"""
        issues = {
            'missing_dependencies': [],
            'circular_dependencies': [],
            'unregistered_services': []
        }
        
        # Check for missing dependencies
        for service_name, dependencies in self.dependency_graph.items():
            for dep in dependencies:
                if dep not in self.registrations and dep not in self.instances:
                    issues['missing_dependencies'].append(f"{service_name} -> {dep}")
        
        # Check for circular dependencies (simplified check)
        visited = set()
        rec_stack = set()
        
        def has_cycle(service: str) -> bool:
            visited.add(service)
            rec_stack.add(service)
            
            for dep in self.dependency_graph.get(service, []):
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    issues['circular_dependencies'].append(f"Cycle involving {service} -> {dep}")
                    return True
            
            rec_stack.remove(service)
            return False
        
        for service in self.registrations:
            if service not in visited:
                has_cycle(service)
        
        return issues
    
    def get_dependency_graph(self) -> Dict[str, Any]:
        """Get visual representation of dependency graph"""
        graph = {}
        
        for service_name, registration in self.registrations.items():
            graph[service_name] = {
                'implementation': registration.implementation.__name__ if registration.implementation else 'Factory',
                'singleton': registration.singleton,
                'dependencies': registration.dependencies or [],
                'resolved': service_name in self.instances
            }
        
        return graph
    
    def dispose(self):
        """Dispose of all singleton instances"""
        disposed_count = 0
        
        for service_name, instance in self.instances.items():
            try:
                if hasattr(instance, 'dispose'):
                    instance.dispose()
                disposed_count += 1
            except Exception as e:
                logger.warning(f"Error disposing {service_name}: {e}")
        
        self.instances.clear()
        logger.info(f"Disposed {disposed_count} service instances")

class Injectable:
    """
    Base class for services that can be injected
    Provides automatic dependency resolution
    """
    
    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses for injection"""
        super().__init_subclass__(**kwargs)
        
        # Auto-registration could be implemented here
        # For now, manual registration is required for better control

# Decorators for easy service registration
def service(interface: Type[T], singleton: bool = True):
    """Decorator to register a service"""
    def decorator(implementation: Type[T]):
        container.register(interface, implementation, singleton)
        return implementation
    return decorator

def singleton(interface: Type[T]):
    """Decorator to register a singleton service"""
    return service(interface, singleton=True)

def transient(interface: Type[T]):
    """Decorator to register a transient service"""
    return service(interface, singleton=False)

# Global service container - single source of truth for all services
container = ServiceContainer()