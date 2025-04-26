#!/usr/bin/env python3
"""
Config system validation tool.
This script checks syntax, imports, consistency and relationships
between files in the configuration system.
"""

import os
import sys
import importlib
import inspect
import pkgutil
import logging
import traceback
import ast
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional
from enum import Enum  # Added import for Enum

# Set up logging with explicit encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('config_validation.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger("config_validator")

# Project root directory
BASE_DIR = Path(__file__).parent

# List of files to check
CONFIG_FILES = [
    "config/system_config.py",
    "config/logging_config.py",
    "config/security_config.py",
    "config/env.py",
    "config/constants.py",
    "config/utils/encryption.py",
    "config/utils/validators.py",
]

# Validation issues
class ValidationResult:
    """Validation result for a file."""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.syntax_errors: List[str] = []
        self.import_errors: List[str] = []
        self.circular_dependencies: List[str] = []
        self.naming_issues: List[str] = []
        self.security_issues: List[str] = []
        self.style_issues: List[str] = []
        self.logic_issues: List[str] = []
        self.consistency_issues: List[str] = []
        self.unused_imports: List[str] = []
        self.unused_variables: List[str] = []
        self.success = True
    
    def add_error(self, category: str, message: str) -> None:
        """Add an error to the list."""
        if hasattr(self, category):
            getattr(self, category).append(message)
            self.success = False
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return not self.success
    
    def get_all_errors(self) -> Dict[str, List[str]]:
        """Get all errors."""
        errors = {}
        for attr_name in [
            'syntax_errors', 'import_errors', 'circular_dependencies', 
            'naming_issues', 'security_issues', 'style_issues', 
            'logic_issues', 'consistency_issues', 'unused_imports',
            'unused_variables'
        ]:
            if hasattr(self, attr_name):
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, list):
                    errors[attr_name] = attr_value
        return errors
    
    def get_error_count(self) -> int:
        """Get total error count."""
        count = 0
        for attr_name in [
            'syntax_errors', 'import_errors', 'circular_dependencies', 
            'naming_issues', 'security_issues', 'style_issues', 
            'logic_issues', 'consistency_issues', 'unused_imports',
            'unused_variables'
        ]:
            if hasattr(self, attr_name):
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, list):
                    count += len(attr_value)
        return count
    
    def __str__(self) -> str:
        """Display result as string."""
        if self.success:
            return f"✓ {self.file_path}: No issues found"
        
        result = [f"✗ {self.file_path}: {self.get_error_count()} issues"]
        
        for category, errors in self.get_all_errors().items():
            if errors:
                category_name = category.replace('_', ' ').capitalize()
                result.append(f"  {category_name}:")
                for error in errors:
                    result.append(f"    - {error}")
        
        return "\n".join(result)

class ConfigValidator:
    """
    Configuration validation tool.
    Checks syntax, imports, dependencies and consistency of config files.
    """
    
    def __init__(self, config_files: List[str]):
        self.config_files = config_files
        self.results: Dict[str, ValidationResult] = {}
        self.import_graph: Dict[str, Set[str]] = {}
        self.module_map: Dict[str, str] = {}  # module_name -> file_path
    
    def validate_all(self) -> Dict[str, ValidationResult]:
        """Validate all configuration files."""
        logger.info("Starting configuration validation...")
        
        for file_path in self.config_files:
            full_path = os.path.join(BASE_DIR, file_path)
            if not os.path.exists(full_path):
                logger.warning(f"File does not exist: {file_path}")
                continue
            
            result = ValidationResult(file_path)
            self.results[file_path] = result
            
            logger.info(f"Checking {file_path}...")
            
            # Check Python syntax
            self.check_syntax(file_path, result)
            
            # If no syntax errors, continue with other checks
            if not result.syntax_errors:
                # Build import graph
                self.build_import_graph(file_path, result)
                
                # Check other issues
                self.check_style(file_path, result)
                self.check_security(file_path, result)
                self.check_logic(file_path, result)
        
        # Check circular dependencies
        self.check_circular_dependencies()
        
        # Check consistency between files
        self.check_consistency()
        
        logger.info("Configuration validation complete")
        return self.results
    
    def check_syntax(self, file_path: str, result: ValidationResult) -> None:
        """Check Python syntax."""
        full_path = os.path.join(BASE_DIR, file_path)
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check syntax
            ast.parse(content, filename=file_path)
            
            # Extract module_name from file_path
            module_path = file_path.replace('/', '.').replace('.py', '')
            self.module_map[module_path] = file_path
            
        except SyntaxError as e:
            result.add_error('syntax_errors', f"Line {e.lineno}: {e.msg}")
        except Exception as e:
            result.add_error('syntax_errors', f"Error checking syntax: {str(e)}")
    
    def build_import_graph(self, file_path: str, result: ValidationResult) -> None:
        """Build import graph and check imports."""
        full_path = os.path.join(BASE_DIR, file_path)
        module_path = file_path.replace('/', '.').replace('.py', '')
        self.import_graph[module_path] = set()
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content, filename=file_path)
            
            # Find imports
            imported_modules = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imported_modules.add(name.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:  # Could be None in case of "from . import x"
                        imported_modules.add(node.module)
            
            # Find project imports
            for module in imported_modules:
                # Check if module is in the project
                if module.startswith('config.'):
                    self.import_graph[module_path].add(module)
            
            # Check for import errors
            for module in imported_modules:
                if module.startswith('config.'):
                    try:
                        # Check if module can be imported
                        importlib.import_module(module)
                    except ImportError as e:
                        result.add_error('import_errors', f"Cannot import module '{module}': {str(e)}")
        
        except Exception as e:
            result.add_error('import_errors', f"Error checking imports: {str(e)}")
    
    def check_circular_dependencies(self) -> None:
        """Check for circular dependencies."""
        
        def find_cycles(node: str, path: List[str], visited: Set[str]) -> None:
            """Find cycles in directed graph."""
            if node in path:
                # Found a cycle
                cycle = path[path.index(node):] + [node]
                file_path = self.module_map.get(cycle[0], cycle[0])
                result = self.results.get(file_path)
                if result:
                    cycle_str = " -> ".join(cycle)
                    result.add_error('circular_dependencies', f"Circular dependency: {cycle_str}")
                return
            
            if node in visited:
                return
            
            visited.add(node)
            path.append(node)
            
            for neighbor in self.import_graph.get(node, []):
                find_cycles(neighbor, path, visited)
            
            path.pop()
        
        # Find cycles from each node
        for node in self.import_graph:
            find_cycles(node, [], set())
    
    def check_style(self, file_path: str, result: ValidationResult) -> None:
        """Check style issues."""
        full_path = os.path.join(BASE_DIR, file_path)
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Check docstrings
            tree = ast.parse(content, filename=file_path)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                    if not ast.get_docstring(node):
                        if isinstance(node, ast.Module):
                            result.add_error('style_issues', "Module has no docstring")
                        elif isinstance(node, ast.ClassDef):
                            result.add_error('style_issues', f"Class '{node.name}' has no docstring")
                        elif isinstance(node, ast.FunctionDef):
                            if not node.name.startswith('_'):  # Skip private methods
                                result.add_error('style_issues', f"Function '{node.name}' has no docstring")
            
            # Check line length (>100 chars)
            for i, line in enumerate(lines, 1):
                if len(line) > 100:
                    result.add_error('style_issues', f"Line {i} too long ({len(line)} chars)")
            
            # Check for import *
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for name in node.names:
                        if name.name == '*':
                            result.add_error('style_issues', f"Line {node.lineno}: Using 'import *' is not recommended")
            
            # Check variable and function naming
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Classes should use PascalCase
                    if not node.name[0].isupper() or '_' in node.name:
                        result.add_error('naming_issues', f"Line {node.lineno}: Class name '{node.name}' doesn't follow PascalCase")
                
                elif isinstance(node, ast.FunctionDef):
                    # Functions should use snake_case
                    if not node.name.islower() and not node.name.startswith('_'):
                        result.add_error('naming_issues', f"Line {node.lineno}: Function name '{node.name}' doesn't follow snake_case")
        
        except Exception as e:
            result.add_error('style_issues', f"Error checking style: {str(e)}")
    
    def check_security(self, file_path: str, result: ValidationResult) -> None:
        """Check security issues."""
        full_path = os.path.join(BASE_DIR, file_path)
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=file_path)
            
            # Check for unsafe imports
            unsafe_modules = ['pickle', 'marshal', 'eval', 'exec']
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        if name.name in unsafe_modules:
                            result.add_error('security_issues', f"Line {node.lineno}: Unsafe module import '{name.name}'")
                elif isinstance(node, ast.ImportFrom):
                    if node.module in unsafe_modules:
                        result.add_error('security_issues', f"Line {node.lineno}: Import from unsafe module '{node.module}'")
            
            # Check for eval() or exec() usage
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in ['eval', 'exec']:
                        result.add_error('security_issues', f"Line {node.lineno}: Unsafe usage of {node.func.id}()")
            
            # Check for hardcoded passwords/API keys
            suspicious_var_names = ['password', 'secret', 'key', 'token', 'api_key', 'api_secret']
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            name = target.id.lower()
                            if any(sus in name for sus in suspicious_var_names):
                                if isinstance(node.value, ast.Str) and len(node.value.s) > 5:
                                    result.add_error('security_issues', f"Line {node.lineno}: Possible hardcoded credential '{target.id}'")
        
        except Exception as e:
            result.add_error('security_issues', f"Error checking security: {str(e)}")
    
    def check_logic(self, file_path: str, result: ValidationResult) -> None:
        """Check logic issues."""
        full_path = os.path.join(BASE_DIR, file_path)
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=file_path)
            
            # Find unused variables
            defined_vars = {}
            used_vars = set()
            
            # Find all defined variables
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            defined_vars[target.id] = node.lineno
            
            # Find all used variables
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    used_vars.add(node.id)
            
            # Find defined but unused variables (except constants)
            for var, lineno in defined_vars.items():
                if var not in used_vars and not var.isupper() and not var.startswith('_'):
                    result.add_error('unused_variables', f"Line {lineno}: Variable '{var}' defined but not used")
            
            # Find unused imports
            imported_modules = {}
            used_modules = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imported_name = name.asname if name.asname else name.name
                        imported_modules[imported_name] = (node.lineno, name.name)
                elif isinstance(node, ast.ImportFrom):
                    for name in node.names:
                        imported_name = name.asname if name.asname else name.name
                        imported_modules[imported_name] = (node.lineno, f"{node.module}.{name.name}" if node.module else name.name)
            
            # Find all used modules
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    if node.id in imported_modules:
                        used_modules.add(node.id)
                elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                    if isinstance(node.value, ast.Name):
                        if node.value.id in imported_modules:
                            used_modules.add(node.value.id)
            
            # Report unused imports
            for module, (lineno, name) in imported_modules.items():
                # Skip typing imports as they're often used for type hints
                if module not in used_modules and not name.startswith('typing'):
                    result.add_error('unused_imports', f"Line {lineno}: Import '{name}' not used")
        
        except Exception as e:
            result.add_error('logic_issues', f"Error checking logic: {str(e)}")
    
    def check_consistency(self) -> None:
        """Check consistency between files."""
        
        # Check configuration consistency
        try:
            # Check SystemConfig
            sys_config_path = "config/system_config.py"
            if sys_config_path in self.results:
                # Check if SystemConfig exists
                module_path = sys_config_path.replace('/', '.').replace('.py', '')
                try:
                    module = importlib.import_module(module_path)
                    if not hasattr(module, 'SystemConfig'):
                        self.results[sys_config_path].add_error(
                            'consistency_issues', "Missing SystemConfig class in system_config.py"
                        )
                except Exception as e:
                    self.results[sys_config_path].add_error(
                        'consistency_issues', f"Error importing module: {str(e)}"
                    )
            
            # Check LoggingConfig
            logging_config_path = "config/logging_config.py"
            if logging_config_path in self.results:
                module_path = logging_config_path.replace('/', '.').replace('.py', '')
                try:
                    module = importlib.import_module(module_path)
                    if not hasattr(module, 'LoggingConfig'):
                        self.results[logging_config_path].add_error(
                            'consistency_issues', "Missing LoggingConfig class in logging_config.py"
                        )
                    
                    # Check if it uses SystemConfig
                    with open(os.path.join(BASE_DIR, logging_config_path), 'r', encoding='utf-8') as f:
                        content = f.read()
                    if 'system_config' not in content:
                        self.results[logging_config_path].add_error(
                            'consistency_issues', "Logging config should use system_config for configuration"
                        )
                except Exception as e:
                    self.results[logging_config_path].add_error(
                        'consistency_issues', f"Error importing module: {str(e)}"
                    )
            
            # Check constants
            constants_path = "config/constants.py"
            if constants_path in self.results:
                module_path = constants_path.replace('/', '.').replace('.py', '')
                try:
                    module = importlib.import_module(module_path)
                    # Check for Enums
                    has_enum = False
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) and issubclass(obj, Enum):
                            has_enum = True
                            break
                    
                    if not has_enum:
                        self.results[constants_path].add_error(
                            'consistency_issues', "Constants.py should use Enum for constant definitions"
                        )
                except Exception as e:
                    self.results[constants_path].add_error(
                        'consistency_issues', f"Error importing module: {str(e)}"
                    )
            
            # Check encryption utils
            encryption_path = "config/utils/encryption.py"
            if encryption_path in self.results:
                module_path = encryption_path.replace('/', '.').replace('.py', '')
                try:
                    module = importlib.import_module(module_path)
                    # Check required functions
                    required_functions = ['encrypt_data', 'decrypt_data']
                    for func in required_functions:
                        if not hasattr(module, func):
                            self.results[encryption_path].add_error(
                                'consistency_issues', f"Missing function {func} in encryption.py"
                            )
                except Exception as e:
                    self.results[encryption_path].add_error(
                        'consistency_issues', f"Error importing module: {str(e)}"
                    )
            
            # Check validators
            validators_path = "config/utils/validators.py"
            if validators_path in self.results:
                module_path = validators_path.replace('/', '.').replace('.py', '')
                try:
                    module = importlib.import_module(module_path)
                    # Check for validation functions
                    validation_functions_count = 0
                    for name, obj in inspect.getmembers(module):
                        if name.startswith('is_valid_') and inspect.isfunction(obj):
                            validation_functions_count += 1
                    
                    if validation_functions_count < 5:
                        self.results[validators_path].add_error(
                            'consistency_issues', f"Insufficient validation functions in validators.py (only {validation_functions_count})"
                        )
                except Exception as e:
                    self.results[validators_path].add_error(
                        'consistency_issues', f"Error importing module: {str(e)}"
                    )
            
        except Exception as e:
            logger.error(f"Error checking consistency: {str(e)}")
    
    def display_results(self) -> None:
        """Display validation results."""
        print("\n=== CONFIGURATION VALIDATION RESULTS ===\n")
        
        success_count = 0
        error_count = 0
        
        for file_path, result in self.results.items():
            print(result)
            print()
            
            if result.success:
                success_count += 1
            else:
                error_count += 1
        
        print(f"Total: {len(self.results)} files checked, {success_count} successful, {error_count} with issues")
    
    def run(self) -> bool:
        """Run the entire validation process and display results."""
        self.validate_all()
        self.display_results()
        
        # Return True if no files have errors
        return all(result.success for result in self.results.values())

def main():
    """Main function to run configuration validation."""
    validator = ConfigValidator(CONFIG_FILES)
    success = validator.run()
    
    if success:
        logger.info("All configuration files passed validation!")
        return 0
    else:
        logger.warning("Some configuration files have issues. Please check and fix.")
        return 1

if __name__ == "__main__":
    sys.exit(main())