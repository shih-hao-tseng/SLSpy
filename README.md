# SLSpy: Python-based Simulation Framework for System Level Synthesis

## Synopsis
*SLSpy* provides a Python-based framework to simulate model-based control systems, especially for system level synthesis (SLS) methods. The simulation workflow follows the descriptions in the paper

Shih-Hao Tseng and James Anderson, ``Deployment Architectures for Cyber-Physical Control Systems,'' 2019.

## System Requirement
* Python 2.7 or higher
* Python pip

For Python 3 or higher version, the user might need to install cvxpy manually. The current pip install method may fail.  

## Install
To install SLSpy, one may type

`sudo make install`

which has been tested under Fedora 30.

To install the packages manually, the user should install
* blas-devel/libblas-dev
* lapack-devel/liblapack-dev

first, then reinstall scs package by

`pip install --no-cache-dir --ignore-installed scs`

or

`pip install --no-cache-dir -I scs`