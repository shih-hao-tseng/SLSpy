# SLSpy: Python-based Simulation Framework for System Level Synthesis

## Synopsis
*SLSpy* provides a Python-based framework to simulate model-based control systems, especially for system level synthesis (SLS) methods. The simulation workflow follows the descriptions in the paper

Shih-Hao Tseng and James Anderson, ``[Deployment Architectures for Cyber-Physical Control Systems](https://arxiv.org/abs/1911.01510),'' 2019.

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

## SLSpy Wiki
A simple wiki page with basic explanation can be found at 
http://slswiki.cms.caltech.edu/index.php/SLSpy

## Built-in Synthesis Algorithms
* System Level Synthesis (FIR version), c.f.\
  James Anderson, John C. Doyle, Steven H. Low, and Nikolai Matni,\
  ``System Level Synthesis,'' *Annual Reviews in Control*, 2019.

  Yuh-Shyang Wang, Nikolai Matni, and John C. Doyle,\
  ``A System Level Approach to Controller Synthesis,'' *IEEE Trans. Autom. Control*, 2019

* Input-Output Parameterization (FIR version), c.f.\
  Luca Furieri, Yang Zheng, Antonis Papachristodoulou, and Maryam Kamgarpour,\
  ``An Input-Output Parameterization of Stabilizing Controllers: Amidst Youla and System Level Synthesis,'' *IEEE Control Systems Letters*, 2019