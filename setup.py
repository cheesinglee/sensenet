"""Setup for package sensenet
"""

from setuptools import setup, find_packages
from os import path, getcwd
from subprocess import check_output

from sensenet import __version__

name = 'sensenet'
base_version = 'andromeda'

git_short = 7

try:
    head = check_output('git rev-parse --short=%d HEAD 2>/dev/null' %
                        git_short, shell=True).strip()

    version = base_version + '-' + head
except:
    version = path.basename(getcwd())[len(name)+1:]

    if version == '':
        version == __version__

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
    setup_requires=['nose'],
    test_suite='nose.collector',
    install_requires=[
        'numpy>=1.17.2,<1.18',
        'tensorflow>=1.14,<1.15',
        'pillow>=6.1,<6.2',
        'keras>=2.3,<2.4'
    ],
    entry_points={
        'console_scripts': [
            'sensenet_write = sensenet.command_line:main'
        ]
    })
