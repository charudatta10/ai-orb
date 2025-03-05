import ast
import io
import sys
import re
import queue
import threading
import signal
import traceback

class SecureSandbox:
    def __init__(self, 
                 max_time=5,  # Maximum execution time in seconds
                 max_output_size=10 * 1024):  # 10 KB output limit
        """
        Initialize a secure sandbox for code execution.
        
        Args:
            max_time (int): Maximum execution time in seconds
            max_output_size (int): Maximum output size in bytes
        """
        self.max_time = max_time
        self.max_output_size = max_output_size

    class SecurityVisitor(ast.NodeVisitor):
        """
        AST Visitor to check for potentially dangerous operations
        """
        BLOCKED_MODULES = {
            'os', 'subprocess', 'sys', 'platform', 
            'ctypes', 'threading', 'multiprocessing', 
            'signal', 'socket', 'fcntl', 'winreg'
        }
        
        BLOCKED_FUNCTIONS = {
            'eval', 'exec', 'compile', 'open', 'input', 
            'breakpoint', '__import__', 'exit', 'quit'
        }
        
        def visit_Import(self, node):
            for alias in node.names:
                if alias.name in self.BLOCKED_MODULES:
                    raise ValueError(f"Import of '{alias.name}' is not allowed")
        
        def visit_ImportFrom(self, node):
            if node.module in self.BLOCKED_MODULES:
                raise ValueError(f"Import from '{node.module}' is not allowed")
        
        def visit_Call(self, node):
            # Check for dangerous function calls
            if isinstance(node.func, ast.Name):
                if node.func.id in self.BLOCKED_FUNCTIONS:
                    raise ValueError(f"Call to '{node.func.id}' is not allowed")
            
            # Prevent accessing private/magic methods
            if isinstance(node.func, ast.Attribute):
                if node.func.attr.startswith('__'):
                    raise ValueError("Access to private methods is not allowed")
            
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
            visitor = self.SecurityVisitor()
            visitor.visit(tree)
        except SyntaxError as e:
            raise ValueError(f"Syntax error: {e}")

    def execute(self, code, global_context=None):
        """
        Execute code in a controlled, secure environment.
        
        Args:
            code (str): Python code to execute
            global_context (dict, optional): Global variables to provide to the code
        
        Returns:
            dict: Execution result with 'output', 'error', and 'success' keys
        """
        # Validate code before execution
        try:
            self.validate_code(code)
        except ValueError as security_error:
            return {
                'success': False,
                'error': str(security_error),
                'output': ''
            }

        # Prepare execution context
        if global_context is None:
            global_context = {}
        local_context = {}

        # Queues for communication between threads
        output_queue = queue.Queue()
        error_queue = queue.Queue()
        result_queue = queue.Queue()

        def run_code():
            # Capture output
            output = io.StringIO()
            error_output = io.StringIO()

            try:
                # Redirect stdout and stderr
                sys.stdout = output
                sys.stderr = error_output

                # Execute the code
                exec(code, global_context, local_context)
                
                # Put successful result
                result_queue.put({
                    'success': True,
                    'output': output.getvalue(),
                    'error': ''
                })
            except Exception as e:
                # Capture any runtime errors
                result_queue.put({
                    'success': False,
                    'output': output.getvalue(),
                    'error': f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
                })
            finally:
                # Restore stdout and stderr
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__

        # Create and start the thread
        exec_thread = threading.Thread(target=run_code)
        exec_thread.daemon = True
        exec_thread.start()

        # Wait for the specified time
        exec_thread.join(timeout=self.max_time)

        # Check if thread is still alive (timeout occurred)
        if exec_thread.is_alive():
            return {
                'success': False,
                'error': f'Code execution exceeded {self.max_time} second time limit',
                'output': ''
            }

        # Retrieve the result
        result = result_queue.get()

        # Truncate output if too large
        if len(result['output']) > self.max_output_size:
            result['output'] = result['output'][:self.max_output_size] + '\n[Output truncated]'

        return result

# Example usage
def run_sandbox_example():
    sandbox = SecureSandbox()
    
    # Safe code example
    safe_code = """
print('Hello, Sandbox!')
x = 5 + 3
print(f'Result: {x}')
"""
    result = sandbox.execute(safe_code)
    print("Safe Code Result:", result)

    # Potentially dangerous code example
    dangerous_code = """
import os
print(os.system('ls'))
"""
    result = sandbox.execute(dangerous_code)
    print("Dangerous Code Result:", result)

if __name__ == '__main__':
    run_sandbox_example()