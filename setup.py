#!/usr/bin/python3
import setuptools
from   amrlib import __version__

# To create the pypi distribution and upload to pypi do..
#   ./setup.py sdist bdist_wheel
#   twine upload dist/*
# To install to the user account d0..
#   ./setup.py install --user

# Load the README.md to use as the long description that shows up on pypi
with open('README.md', 'r') as fh:
    readme = fh.read()
    # Remove lines with png file references (because these don't seem to display on pypi)
    lines = readme.splitlines()
    lines = [l for l in lines if not '.png' in l]
    readme = '\n'.join(lines)


setuptools.setup(
    name='amrlib',
    version=__version__,
    author='Brad Jascob',
    author_email='bjascob@msn.com',
    description='A python library that makes AMR parsing, generation and visualization simple.',
    long_description=readme,
    long_description_content_type='text/markdown',
    url='https://github.com/bjascob/amrlib',
    # The following adds data files for the binary distribution only (not the source)
    # This impacts `setup bdist_wheel`.  Use the MANIFEST.in file to add data files to
    # the source package.  Also note that just using wildcards (ie.. *.csv) without the
    # path doesn't work unless there's an __init__.py in the directory because setup
    # doesn't look in there without it.
    include_package_data=True,
    package_data={'amrlib':['amr_view/*',
                            'alignments/faa_aligner/model_aligner_faa.tar.gz',
                            'alignments/faa_aligner/resources/*.txt',
                            'alignments/isi_hand_alignments/*.txt']},
    packages=setuptools.find_packages(),
    # Minimal requirements here.  More extensive list in requirements.txt
    install_requires=['penman>=1.1.0', 'torch>=1.6', 'numpy', 'spacy>=2.0', 'tqdm', 'transformers>=3.0', 'smatch'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        "Operating System :: OS Independent",
    ],
    # Scripts to be packaged and installed in an exe directory
    entry_points={ "gui_scripts": ['amr_view = amrlib.amr_view.cli:main' ]}
)
