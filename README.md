# SLSpy: Python-based Simulation Framework for System Level Synthesis

## Synopsis
*SLSpy* provides a Python-based framework to simulate model-based control systems, especially for system level synthesis (SLS) methods.

## System Requirement
* Python 2.7 or higher
* Python pip

For Python 3 or higher version, the user might need to install cvxpy manually. The current pip install method may fail.  

## Install
To install necessary packages, one may type

`make install`

which has been tested under Fedora 30. Ubuntu support will be available soon.

To install the packages manually, the user should install
* blas-devel/libblas-dev
* lapack-devel/liblapack-dev

first, then reinstall scs package by

`pip install --no-cache-dir --ignore-installed scs`

or

`pip install --no-cache-dir -I scs`