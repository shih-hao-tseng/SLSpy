#!/usr/bin/env python

"Setuptools params"

from setuptools import setup, find_packages

# Get version number from source tree
from slspy import VERSION

modname = distname = 'slspy'

setup(
    name=distname,
    version=VERSION,
    description='Python-based Simulation Framework for System Level Synthesis',
    url='https://github.com/shih-hao-tseng/SLSpy',
    author='Shih-Hao Tseng',
    author_email='shtseng@caltech.edu',
    packages=[ 'slspy', 'slspy.examples' ],
    long_description="""
        SLSpy provides a Python-based framework to simulate
        model-based control systems, especially for system
        level synthesis (SLS) methods.
        """,
    classifiers=[
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          "Programming Language :: Python",
          "Development Status :: 4 - Beta",
          "Intended Audience :: Developers",
          "Topic :: System",
    ],
    keywords='SLS',
    license='GPLv3',
    install_requires=[
        'setuptools',
        'numpy',
        'cvxpy',
        'matplotlib'
    ],
    python_requires='>=2.7'
)
