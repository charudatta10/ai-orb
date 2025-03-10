from typing import Any, Callable, Dict, Optional
import inspect

class LLMTool:
    """
    A wrapper class for creating callable tools compatible with Language Learning Models.
    
    This class allows easy conversion of standard Python functions into LLM-compatible tools
    with additional metadata and flexible input/output handling.
    
    Attributes:
        func (Callable): The underlying function to be wrapped
        name (str): Name of the tool
        description (str): Detailed description of the tool's purpose and usage
    """
    
    def __init__(
        self, 
        func: Callable, 
        name: Optional[str] = None, 
        description: Optional[str] = None
    ):
        """
        Initialize the LLMTool wrapper.
        
        Args:
            func (Callable): The function to be wrapped
            name (Optional[str]): Custom name for the tool. Defaults to function name if not provided.
            description (Optional[str]): Custom description for the tool. 
                                         Defaults to function's docstring if not provided.
        """
        self.func = func
        self.name = name or func.__name__
        self.description = description or (func.__doc__ or "No description provided")
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Allow direct calling of the tool with arguments.
        
        Provides a direct method to invoke the wrapped function.
        
        Args:
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
        
        Returns:
            The result of the function call
        """
        return self.forward(*args, **kwargs)
    
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """
        Forward method to execute the wrapped function.
        
        This method provides additional flexibility in argument handling and 
        potential preprocessing or postprocessing of inputs/outputs.
        
        Args:
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
        
        Returns:
            The result of the function call
        """
        # Extract function signature to handle different input scenarios
        sig = inspect.signature(self.func)
        try:
            # Bind arguments to ensure compatibility
            bound_arguments = sig.bind(*args, **kwargs)
            bound_arguments.apply_defaults()
            
            # Call the function with bound arguments
            return self.func(*bound_arguments.args, **bound_arguments.kwargs)
        except TypeError as e:
            raise ValueError(f"Error calling tool {self.name}: {str(e)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the tool to a dictionary representation.
        
        Useful for serialization or LLM tool registration.
        
        Returns:
            Dict containing tool metadata and function information
        """
        return {
            "name": self.name,
            "description": self.description,
            "callable": self.forward
        }


# Demonstrate usage
if __name__ == "__main__":
    ...
