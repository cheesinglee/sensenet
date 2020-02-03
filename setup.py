"""Setup for package sensenet
"""

from os import path
from setuptools import setup, find_packages

from sensenet import __version__

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.md'), 'r') as f:
    long_description = f.read()

setup(
    name='sensenet',
    version=version,
    author = 'BigML Team',
    author_email = 'team@bigml.com',
    url = 'http://bigml.com/',
    description='Network builder for laminar topologies',
    long_description=long_description,
    packages=find_packages(),
    tests_require=[
        'nose>=1.3, 1.4',
        'pillow>=6.1,<6.2'
    ],
    test_suite='nose.collector',
    install_requires=[
        'numpy>=1.17.2,<1.18',
        'tensorflow>=2.1,<2.2',
        'keras>=2.3,<2.4'
    ],
    entry_points={
        'console_scripts': [
            'sensenet_write = sensenet.command_line:main'
        ]
    })
