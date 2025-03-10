import ast
import io
import sys
import re
import queue
import threading
import signal
import traceback
import builtins
import importlib

from ai_orb.tool import LLMTool

class SecureSandbox:
    def __init__(self, 
                 max_time=5,  # Maximum execution time in seconds
                 max_output_size=10 * 1024,  # 10 KB output limit
                 allowed_modules=None):  # Explicitly allowed modules
        """
        Initialize a secure sandbox for code execution with enhanced security controls.
        
        Args:
            max_time (int): Maximum wall-clock execution time in seconds
            max_output_size (int): Maximum output size in bytes
            allowed_modules (list): List of explicitly allowed modules
        """
        self.max_time = max_time
        self.max_output_size = max_output_size
        self.allowed_modules = set(allowed_modules or [])

    class SecurityVisitor(ast.NodeVisitor):
        """
        AST Visitor to check for potentially dangerous operations
        """
        def __init__(self, allowed_modules=None):
            self.allowed_modules = set(allowed_modules or [])
            # Define blocked modules
            self.BLOCKED_MODULES = {
                'os', 'subprocess', 'sys', 'platform', 
                'ctypes', 'threading', 'multiprocessing', 
                'signal', 'socket', 'fcntl', 'winreg',
                'shutil', 'pathlib', 'pty', 'tempfile'
            }
            # Remove explicitly allowed modules from blocked list
            self.BLOCKED_MODULES -= self.allowed_modules
            
            # Define blocked functions
            self.BLOCKED_FUNCTIONS = {
                'eval', 'exec', 'compile', 'open', 'input', 
                'breakpoint', '__import__', 'exit', 'quit'
            }
        
        def visit_Import(self, node):
            for alias in node.names:
                module_parts = alias.name.split('.')
                base_module = module_parts[0]
                if base_module in self.BLOCKED_MODULES:
                    raise ValueError(f"Import of '{alias.name}' is not allowed")
            self.generic_visit(node)
        
        def visit_ImportFrom(self, node):
            if node.module:
                module_parts = node.module.split('.')
                base_module = module_parts[0]
                if base_module in self.BLOCKED_MODULES:
                    raise ValueError(f"Import from '{node.module}' is not allowed")
            self.generic_visit(node)
        
        def visit_Call(self, node):
            # Check for dangerous function calls
            if isinstance(node.func, ast.Name):
                if node.func.id in self.BLOCKED_FUNCTIONS:
                    raise ValueError(f"Call to '{node.func.id}' is not allowed")
            
            # Prevent accessing private/magic methods
            if isinstance(node.func, ast.Attribute):
                if node.func.attr.startswith('__') and node.func.attr not in ('__len__', '__str__', '__repr__'):
                    raise ValueError(f"Access to '{node.func.attr}' is not allowed")
            
            # Continue traversing the AST
            self.generic_visit(node)

    def validate_code(self, code):
        """
        Validate the code by checking for security risks using AST.
        
        Args:
            code (str): Python code to validate
        
        Raises:
            ValueError: If potentially dangerous code is detected
        """
        # Remove encoding declarations and shebang lines
        code = re.sub(r'^#!.*\n?', '', code)
        code = re.sub(r'^.*coding[:=]\s*[a-zA-Z0-9\-]+.*\n?', '', code)
        
        try:
            tree = ast.parse(code)
            visitor = self.SecurityVisitor(allowed_modules=self.allowed_modules)
            visitor.visit(tree)
        except SyntaxError as e:
            raise ValueError(f"Syntax error: {e}")

    def execute(self, func_or_code, *args, **kwargs):
        """
        Execute function or code in a controlled, secure environment.
        
        Args:
            func_or_code: Either a string of Python code or a callable function
            *args: Arguments to pass to the function (if func_or_code is callable)
            **kwargs: Keyword arguments to pass to the function (if func_or_code is callable)
        
        Returns:
            If func_or_code is a string: dict with 'output', 'error', 'success'
            If func_or_code is callable: a wrapper function that executes in the sandbox
        """
        # Check if input is a function
        if callable(func_or_code):
            # Return a wrapper function
            def secure_wrapper(*wrapper_args, **wrapper_kwargs):
                # Combine original args/kwargs with wrapper args/kwargs
                combined_args = args + wrapper_args
                combined_kwargs = {**kwargs, **wrapper_kwargs}
                
                # Execute the function in the sandbox
                return self._execute_function(func_or_code, *combined_args, **combined_kwargs)
            
            return secure_wrapper
        else:
            # Execute code string directly
            return self._execute_code(func_or_code)

    def _create_safe_globals(self):
        """Create a safe globals dictionary with controlled imports."""
        safe_globals = {}
        
        # Add built-in functions and types (excluding blocked ones)
        for name in dir(builtins):
            if name not in self.SecurityVisitor().BLOCKED_FUNCTIONS:
                safe_globals[name] = getattr(builtins, name)
        
        # Add a secure import function
        def secure_import(name, *args, **kwargs):
            if name in self.allowed_modules:
                return importlib.import_module(name)
            else:
                raise ImportError(f"Import of '{name}' is not allowed")
        
        # Add import function to globals
        safe_globals['__import__'] = secure_import
        
        # Pre-import allowed modules
        for module_name in self.allowed_modules:
            try:
                safe_globals[module_name] = importlib.import_module(module_name)
            except ImportError:
                pass  # Skip modules that can't be imported
        
        return safe_globals

    def _execute_code(self, code):
        """Execute code string in the sandbox."""
        # Validate code
        try:
            self.validate_code(code)
        except ValueError as security_error:
            return {
                'success': False,
                'error': str(security_error),
                'output': ''
            }
        
        # Create execution environments
        safe_globals = self._create_safe_globals()
        local_vars = {}
        
        # Queues for communication
        output_queue = queue.Queue()
        result_queue = queue.Queue()
        
        def run_code():
            # Set up output capturing
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            old_stdout, old_stderr = sys.stdout, sys.stderr
            
            try:
                # Redirect output
                sys.stdout, sys.stderr = stdout_capture, stderr_capture
                
                # Execute the code
                exec(code, safe_globals, local_vars)
                
                # Put successful result in queue
                result_queue.put({
                    'success': True,
                    'output': stdout_capture.getvalue(),
                    'error': stderr_capture.getvalue(),
                    'result': None
                })
            except Exception as e:
                # Capture errors
                traceback_str = traceback.format_exc()
                result_queue.put({
                    'success': False,
                    'output': stdout_capture.getvalue(),
                    'error': f"{stderr_capture.getvalue()}\n{str(e)}\n{traceback_str}",
                    'result': None
                })
            finally:
                # Restore stdout/stderr
                sys.stdout, sys.stderr = old_stdout, old_stderr
        
        # Create and start execution thread
        exec_thread = threading.Thread(target=run_code)
        exec_thread.daemon = True
        exec_thread.start()
        
        # Wait for thread to complete with timeout
        exec_thread.join(timeout=self.max_time)
        
        # Check if execution timed out
        if exec_thread.is_alive():
            return {
                'success': False,
                'error': f'Code execution exceeded {self.max_time} second time limit',
                'output': '',
                'result': None
            }
        
        # Get result
        try:
            result = result_queue.get(block=False)
            
            # Truncate output if needed
            if len(result['output']) > self.max_output_size:
                result['output'] = result['output'][:self.max_output_size] + '\n[Output truncated]'
                
            return result
        except queue.Empty:
            return {
                'success': False,
                'error': 'Execution failed to produce a result',
                'output': '',
                'result': None
            }

    def _execute_function(self, func, *args, **kwargs):
        """Execute a function in the sandbox and return its result."""
        # Handle LLMTool objects
        if isinstance(func, LLMTool):
            func = func.func  # Extract the underlying function

        # Create code to call the function
        module_name = func.__module__
        func_name = func.__name__

        # Create code that imports the module and calls the function
        code = f"""
# Import the function's module
import {module_name}

# Call the function with arguments
_result = {module_name}.{func_name}(*_args, **_kwargs)
"""

        # Set up execution environment
        safe_globals = self._create_safe_globals()
        local_vars = {
            '_args': args,
            '_kwargs': kwargs
     }

        # Add the function's module to allowed modules
        self.allowed_modules.add(module_name)

        # Validate and execute
        try:
            self.validate_code(code)
        except ValueError as security_error:
            raise ValueError(f"Function validation failed: {security_error}")

        # Create execution queue
        result_queue = queue.Queue()

        def run_func():
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            old_stdout, old_stderr = sys.stdout, sys.stderr

            try:
                # Redirect output
                sys.stdout, sys.stderr = stdout_capture, stderr_capture

                # Execute function code
                exec(code, safe_globals, local_vars)

                # Get the function result
                result = local_vars.get('_result')

                # Put result in queue
                result_queue.put({
                    'success': True,
                    'output': stdout_capture.getvalue(),
                    'error': stderr_capture.getvalue(),
                    'result': result
                })
            except Exception as e:
                # Capture errors
                traceback_str = traceback.format_exc()
                result_queue.put({
                    'success': False,
                    'output': stdout_capture.getvalue(),
                    'error': f"{stderr_capture.getvalue()}\n{str(e)}\n{traceback_str}",
                    'result': None
             })
            finally:
                # Restore stdout/stderr
                sys.stdout, sys.stderr = old_stdout, old_stderr

        # Create and start execution thread
        exec_thread = threading.Thread(target=run_func)
        exec_thread.daemon = True
        exec_thread.start()

        # Wait for thread to complete with timeout
        exec_thread.join(timeout=self.max_time)

        # Check if execution timed out
        if exec_thread.is_alive():
            raise TimeoutError(f'Function execution exceeded {self.max_time} second time limit')

        # Get result
        try:
            result = result_queue.get(block=False)

            if not result['success']:
                raise RuntimeError(result['error'])

            return result['result']
        except queue.Empty:
            raise RuntimeError('Function execution failed to produce a result')

if __name__ == '__main__':
    ...