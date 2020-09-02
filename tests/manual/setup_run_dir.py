import os
import sys

# Simple script that on import will cause the importee to run from 2 levels up.

# Where we want to trick the system to think we're running from
relative_path = '..' + os.sep + '..'

# Alter the python module search path.  First entry is always the local directory.
sys.path[0] = os.path.abspath(os.path.join(sys.path[0], relative_path))

# Change the working directory too for loading data, logging, etc ...
os.chdir(sys.path[0])
