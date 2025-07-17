# Configuration file for the Sphinx documentation builder.

import os
import sys
import inspect

# At the top of conf.py, after the imports
print(f"Current working directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")
print(f"Python path: {sys.path}")

# Check if EIANN module can be imported
try:
    import EIANN
    print(f"EIANN module found at: {EIANN.__file__}")
except ImportError as e:
    print(f"Could not import EIANN: {e}")

# Add the package to the path
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../EIANN'))

# Project information
project = 'EIANN'
copyright = '2025, Milstein Lab'
author = 'A.R.Galloni, A.D.Milstein'

# Extensions loaded by conf.py (these are in addition to those in _config.yml)
extensions = []

# HTML configuration
html_show_sourcelink = True

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx for linking to other docs
intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# Define linkcode_resolve function
def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object for GitHub source links
    """
    if domain != 'py':
        return None
    
    if not info['module']:
        return None
    
    # Get the module
    try:
        module = __import__(info['module'], fromlist=[''])
    except ImportError:
        return None
    
    # Get the object
    obj = module
    for part in info['fullname'].split('.'):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None
    
    # Get the source file and lines
    try:
        source_file = inspect.getsourcefile(obj)
        source_lines = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        return None
    
    if source_file is None:
        return None
    
    # Convert absolute path to relative path from repo root
    try:
        source_file = os.path.normpath(source_file)
        source_file = os.path.abspath(source_file)
        
        # Try to find the relative path from the repository root
        if '/EIANN/' in source_file:
            # Split at the first occurrence of /EIANN/
            parts = source_file.split('/EIANN/', 1)
            rel_path = 'EIANN/' + parts[1]
        elif '\\EIANN\\' in source_file:  # Windows path
            parts = source_file.split('\\EIANN\\', 1)
            rel_path = 'EIANN/' + parts[1].replace('\\', '/')
        else:
            return None
            
    except (IndexError, ValueError):
        return None
    
    # GitHub repository info
    github_user = "Milstein-Lab"
    github_repo = "EIANN"
    github_branch = "main"
    
    # Build the GitHub URL with line numbers
    line_start = source_lines[1]
    line_end = line_start + len(source_lines[0]) - 1
    
    return f"https://github.com/{github_user}/{github_repo}/blob/{github_branch}/{rel_path}#L{line_start}-L{line_end}"

# Make sure the function is available in the global namespace
globals()['linkcode_resolve'] = linkcode_resolve
