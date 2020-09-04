"""Setup for package sensenet
"""

tf_ver = 'tensorflow>=2.3,<2.4'
import_err = 'Cannot import tensorflow.  Please run `pip install %s`' % tf_ver

try:
    import tensorflow as tf
except ModuleNotFoundError:
    raise ImportError(import_err)

from os import path
from setuptools import setup, Extension, find_packages

from sensenet import __version__, __tree_ext_prefix__

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.md'), 'r') as f:
    long_description = f.read()

compile_args = ['-std=c++11', '-fPIC'] + tf.sysconfig.get_compile_flags()

tree_module = Extension(__tree_ext_prefix__,
                        define_macros = [('MAJOR_VERSION', '0'),
                                         ('MINOR_VERSION', '1')],
                        include_dirs=[tf.sysconfig.get_include()],
                        library_dirs=[tf.sysconfig.get_lib()],
                        extra_compile_args=compile_args,
                        extra_link_args=tf.sysconfig.get_link_flags(),
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
        'pillow>=7.2,<7.3',
        'scikit-learn>=0.23,<0.24'
    ],
    test_suite='nose.collector',
    install_requires=[
        'numpy>=1.18,<1.19',
        tf_ver
    ])
