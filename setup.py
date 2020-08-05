"""Setup for package sensenet
"""
import tensorflow as tf

from os import path
from setuptools import setup, Extension, find_packages

from sensenet import __version__, __tree_ext_prefix__

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.md'), 'r') as f:
    long_description = f.read()

tree_module = Extension(__tree_ext_prefix__,
                        define_macros = [('MAJOR_VERSION', '0'),
                                         ('MINOR_VERSION', '1')],
                        include_dirs = [tf.sysconfig.get_include()],
                        library_dirs = [tf.sysconfig.get_lib()],
                        extra_compile_args=['-std=c++11'],
                        sources = ['cpp/tree_op.cc'])
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
    ext_modules=[tree_module],
    tests_require=[
        'nose>=1.3,<1.4',
        'pillow>=6.1,<6.2'
    ],
    test_suite='nose.collector',
    install_requires=[
        'numpy>=1.18,<1.19',
        'tensorflow>=2.3,<2.4',
    ])
