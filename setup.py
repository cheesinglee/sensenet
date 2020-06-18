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
    name='bigml-sensenet',
    version=__version__,
    author = 'BigML Team',
    author_email = 'team@bigml.com',
    url = 'http://bigml.com/',
    description='Network builder for bigml deepnet topologies',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    tests_require=[
        'nose>=1.3,<1.4',
        'pillow>=6.1,<6.2'
    ],
    test_suite='nose.collector',
    install_requires=[
        'numpy>=1.18,<1.19',
        'tensorflow>=2.2,<2.3',
    ])
