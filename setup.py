"""Disentangling Time-series Spectra with Gaussian Processes: Applications to Radial Velocity Analysis

https://github.com/iancze/psoap
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

import numpy as np

# set the command line scripts
entry_points = {'console_scripts': [
    'psoap-initialize = psoap.initialize:main',
    'psoap-sample = psoap.sample:main',
    'psoap-sample-george = psoap.sample_george:main',
    'psoap-sample-parallel = psoap.sample_parallel:main',
    'psoap-plot-samples = psoap.plot_samples:main'
]}


# Required for cython
from setuptools.extension import Extension
from Cython.Build import cythonize

here = path.abspath(path.dirname(__file__))

extensions = [Extension("psoap.matrix_functions", ["psoap/matrix_functions.pyx"],
                  include_dirs=[np.get_include()])]

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='psoap',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.0.1',

    description='Gaussian Processes for Radial Velocity Analysis',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/iancze/PSOAP',

    # Author details
    author='Ian Czekala',
    author_email='iancze@gmail.com',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],

    # What does your project relate to?
    keywords='science astronomy spectra',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['numpy', 'scipy', 'cython', 'h5py', 'emcee', 'astropy', 'matplotlib'],

    entry_points=entry_points,
    # List additional groups of dependencies here (e.g. development
    # dependencies). You can install these using the following syntax,
    # for example:
    # $ pip install -e .[dev,test]
    extras_require={
        'dev': ['check-manifest'],
        'test': ['coverage'],
    },

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    package_data={
        'psoap': ['data/*.yaml',
                'data/*.dat',
                'data/41Dra/*',
                'data/GJ3305AB/*',
                'data/Gl417BC/*',
                'data/Gl570BC/*',
                'data/HD10009/*',
                'data/Sigma248/*'],
    },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/LkCa14.hdf5', 'data/LkCa15.hdf5'])],

    ext_modules = cythonize(extensions),

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    # entry_points={
    #     'console_scripts': [
    #         'sample=sample:main',
    #     ],
    # },
)
