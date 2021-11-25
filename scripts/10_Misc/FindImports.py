#!/usr/bin/python3
import setup_run_dir    # Set the working directory and python sys.path to 2 levels above
import os
import logging
import fnmatch
import pkgutil


logger = logging.getLogger(__name__)


# Recursively find all files matching the pattern
def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename


def find_imports(fpath):
    imset = set()
    with open(fpath) as f:
        lines = f.readlines()
    # strip line-feeds and keep lines with "import" in them
    lines = [l.strip() for l in lines if 'import' in l]
    for line in lines:
        # Make sure it's not a comment line
        if line.startswith('#'):
            continue
        # Skip anything dymatic with importlib
        if 'importlib' in line:
            continue
        # Split words on the line into a list
        parts = line.split()
        # Look for "from X import Y" and add X to the set
        try:
            index = parts.index('from')
            lib = parts[index+1]
            # Skip relative imports since these are always local
            if not lib.startswith('.'):
                # only keep the top level module/home/bjascob/.local/lib/python3.8/site-packages
                modules = lib.split('.')
                imset.add(modules[0])
            continue
        except ValueError:
            pass
        # If the above doesn't work, look for "import X" and add X to the set
        try:
            index = parts.index('import')
            lib = parts[index+1]
            # Skip relative imports since these are always local
            if not lib.startswith('.'):
                # only report the first portion of the name
                modules = lib.split('.')
                imset.add(modules[0])
            continue
        except ValueError:
            pass
        # If neither worked then the line couldn't be parsed correctly
        logger.warning('Not able to parse line %s' % line)
    return imset

# Module info is a named-tuple that has fields: 'count', 'index', 'ispkg', 'module_finder', 'name'
def get_module_info():
    minfo = {}
    for x in pkgutil.iter_modules():
        minfo[x.name] = x
    return minfo

# Get the 3rd party packages
def get_3rd_party(packages):
    plist = [p for p in packages if ('site-packages' in p[1] or 'dist-packages' in p[1])]
    return plist

# Get the package version
def get_package_version(package):
    pkgname = package[0]
    dirname = package[1]
    # Try to version the version from the directory name
    dirs = [d.lower() for d in os.listdir(dirname) if os.path.isdir(os.path.join(dirname, d))]
    ver_name = pkgname.lower() + '-'
    dirs = [d for d in dirs if d.startswith(ver_name)]
    if len(dirs) > 1:
        logger.warning('Found multiple versions: %s' % dirs)
    if len(dirs) == 0:
        logger.warning('No version directory found for %s %s' % (dirname, pkgname))
    # Extract the version
    vstring = dirs[0][len(ver_name):]
    if vstring.endswith('.dist-info'):
        vstring = vstring[:-len('.dist-info')]
    elif vstring.endswith('.egg-info'):
        vstring = vstring[:-len('.egg-info')]
    return vstring



if __name__ == '__main__':
    root_directory = 'amrlib'
    print_all       = False
    print_3rd_party = True
    skips           = set(['amr'])    # amr is part of the smatch package

    # Get all the imports
    imset = set()
    for fn in find_files(root_directory, '*.py'):
        imset |= find_imports(fn)

    # Get all module info and build a list of package locations from it
    packages = []
    minfo = get_module_info()
    for x in sorted(imset):
        mi = minfo.get(x, None)
        if mi is None:
            packages.append( (x, '') )
        else:
            packages.append( (x, mi.module_finder.path) )

    # Print the results - tuple is (name, path)
    packages = sorted(packages, key=lambda x:x[0])
    packages = sorted(packages, key=lambda x:x[1])
    if print_all:
        print('All identified packages / installed locations')
        for package in packages:
            print('%-16s %s' % package)
        print()
    if print_3rd_party:
        print('3rd party installed packages')
        for package in sorted(get_3rd_party(packages)):
            pkg_name = package[0]
            if pkg_name not in skips:
                version  = get_package_version(package)
                print('%-16s version:  %s' % (pkg_name, version))
        print()
