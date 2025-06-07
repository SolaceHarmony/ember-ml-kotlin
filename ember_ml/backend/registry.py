"""
Backend Registry Module

This module provides a centralized registry for backend modules and a proxy module pattern
for dynamic attribute forwarding. This allows for seamless backend switching without
requiring explicit alias updates in each module.
"""

import importlib
import sys
from typing import Dict, List, Set, Callable, Any, Optional, Type, TypeVar, cast

# Type variable for generic proxy class
T = TypeVar('T')

class BackendRegistry:
    """
    Singleton registry for backend modules.

    This class maintains a registry of all modules that need to be updated when the backend changes.
    It also provides methods for registering modules and notifying them of backend changes.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BackendRegistry, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the registry."""
        self._registered_modules: Set[str] = set()
        self._registered_instances: Dict[str, Any] = {}
        self._backend_name: Optional[str] = None
        self._backend_module: Optional[Any] = None
        self._callbacks: List[Callable[[], None]] = []

    def register_module(self, module_name: str, instance: Any = None) -> None:
        """
        Register a module to be notified of backend changes.

        Args:
            module_name: The fully qualified name of the module to register
            instance: The module instance to register (optional)
        """
        self._registered_modules.add(module_name)
        if instance is not None:
            self._registered_instances[module_name] = instance
            # If we already have a backend set, notify the instance immediately
            if self._backend_name is not None and self._backend_module is not None:
                try:
                    if hasattr(instance, '_on_backend_change'):
                        instance._on_backend_change(self._backend_name, self._backend_module)
                except Exception as e:
                    print(f"Error notifying instance {module_name} of backend change: {e}")

    def register_callback(self, callback: Callable[[], None]) -> None:
        """
        Register a callback to be called when the backend changes.

        Args:
            callback: A function to call when the backend changes
        """
        self._callbacks.append(callback)

    def set_backend(self, backend_name: str, backend_module: Any) -> None:
        """
        Set the current backend and notify all registered modules.

        Args:
            backend_name: The name of the backend
            backend_module: The backend module object
        """
        print(f"DEBUG: BackendRegistry.set_backend called with backend_name={backend_name}, backend_module={backend_module.__name__ if backend_module else None}")
        print(f"DEBUG: Registered modules: {self._registered_modules}")
        print(f"DEBUG: Registered instances: {list(self._registered_instances.keys())}")

        if backend_name == self._backend_name:
            print(f"DEBUG: Backend already set to {backend_name}, returning")
            return

        self._backend_name = backend_name
        self._backend_module = backend_module

        # Notify all registered module instances
        for module_name, instance in self._registered_instances.items():
            try:
                print(f"DEBUG: Notifying instance {module_name} of backend change")
                if hasattr(instance, '_on_backend_change'):
                    print(f"DEBUG: Instance {module_name} has _on_backend_change method, calling it")
                    instance._on_backend_change(backend_name, backend_module)
                else:
                    print(f"DEBUG: Instance {module_name} does not have _on_backend_change method")
            except Exception as e:
                print(f"Error notifying instance {module_name} of backend change: {e}")

        # Notify all registered modules (for backward compatibility)
        for module_name in self._registered_modules:
            if module_name not in self._registered_instances:
                try:
                    print(f"DEBUG: Notifying module {module_name} of backend change")
                    module = sys.modules.get(module_name)
                    if module and hasattr(module, '_on_backend_change'):
                        print(f"DEBUG: Module {module_name} has _on_backend_change method, calling it")
                        module._on_backend_change(backend_name, backend_module)
                    else:
                        print(f"DEBUG: Module {module_name} does not have _on_backend_change method or is not in sys.modules")
                except Exception as e:
                    print(f"Error notifying module {module_name} of backend change: {e}")

        # Call all registered callbacks
        for callback in self._callbacks:
            try:
                print(f"DEBUG: Calling callback {callback}")
                callback()
            except Exception as e:
                print(f"Error calling callback for backend change: {e}")

    def get_backend(self) -> tuple[Optional[str], Optional[Any]]:
        """
        Get the current backend name and module.

        Returns:
            A tuple of (backend_name, backend_module)
        """
        return self._backend_name, self._backend_module


class ProxyModule:
    """
    Base class for proxy modules that dynamically forward attribute access to the current backend.

    This class should be subclassed by modules that need to dynamically forward attribute access
    to the current backend. Subclasses should implement the _get_backend_module method to return
    the appropriate backend module.
    """
    def __init__(self, name: str, parent_module: Optional[str] = None):
        """
        Initialize the proxy module.

        Args:
            name: The name of this module
            parent_module: The fully qualified name of the parent module, if any
        """
        self._name = name
        self._parent_module = parent_module
        self._full_name = f"{parent_module}.{name}" if parent_module else name
        self._backend_module = None

        print(f"DEBUG: ProxyModule.__init__ called for {self._full_name}")

        # Register with the registry
        registry = BackendRegistry()
        registry.register_module(self._full_name, self)
        print(f"DEBUG: Registered {self._full_name} with registry as instance")

    def _on_backend_change(self, backend_name: str, backend_module: Any) -> None:
        """
        Called when the backend changes.

        Args:
            backend_name: The name of the new backend
            backend_module: The new backend module
        """
        self._backend_module = self._get_backend_module(backend_name, backend_module)

    def _get_backend_module(self, backend_name: str, backend_module: Any) -> Any:
        """
        Get the backend module for this proxy.

        This method should be overridden by subclasses to return the appropriate backend module.

        Args:
            backend_name: The name of the backend
            backend_module: The main backend module

        Returns:
            The backend module for this proxy
        """
        print(f"DEBUG: _get_backend_module called for {self._full_name} with backend_name={backend_name}, backend_module={backend_module.__name__ if backend_module else None}")
        raise NotImplementedError("Subclasses must implement _get_backend_module")

    def __getattr__(self, name: str) -> Any:
        """
        Dynamically forward attribute access to the current backend module.

        Args:
            name: The name of the attribute to access

        Returns:
            The attribute from the backend module

        Raises:
            AttributeError: If the attribute is not found in the backend module
        """
        if self._backend_module is None:
            raise AttributeError(f"No backend module set for {self._full_name}")

        try:
            return getattr(self._backend_module, name)
        except AttributeError:
            raise AttributeError(f"'{self._full_name}' has no attribute '{name}' in backend {self._backend_module.__name__}")


def create_proxy_module(module_name: str, backend_path_template: str) -> Type[ProxyModule]:
    """
    Create a proxy module class for a specific module.

    This function creates a subclass of ProxyModule that implements the _get_backend_module method
    to return the appropriate backend module based on the backend_path_template.

    Args:
        module_name: The name of the module
        backend_path_template: A template string for the backend module path, e.g., "{backend}.{module}"
            where {backend} will be replaced with the backend name and {module} with the module name

    Returns:
        A subclass of ProxyModule
    """
    class SpecificProxyModule(ProxyModule):
        def _get_backend_module(self, backend_name: str, backend_module: Any) -> Any:
            """Get the backend module for this proxy."""
            # Replace placeholders in the template
            backend_path = backend_path_template.format(
                backend=backend_module.__name__,
                module=module_name
            )

            print(f"DEBUG: SpecificProxyModule._get_backend_module called for {self._full_name} with backend_path={backend_path}")

            try:
                module = importlib.import_module(backend_path)
                print(f"DEBUG: Successfully imported module {backend_path}: {module.__name__}")
                return module
            except ImportError as e:
                print(f"Warning: Could not import backend module {backend_path}: {e}")
                return None

    return SpecificProxyModule
