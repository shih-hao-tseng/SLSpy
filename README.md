# SLSpy: Python-based System-Level Controller Synthesis Framework

## Synopsis
*SLSpy* provides a Python-based framework to design and simulate model-based control systems, especially for system level synthesis (SLS) methods. The details of the framework are described in the paper

Shih-Hao Tseng and Jing Shuang (Lisa) Li, ``SLSpy: Python-Based System-Level Controller Synthesis Framework,'' 2020 (submitted for review).

The synthesis workflow follows the structure in the paper

Shih-Hao Tseng and James Anderson, ``[Deployment Architectures for Cyber-Physical Control Systems](https://arxiv.org/abs/1911.01510),'' 2019.

## System Requirement
* Python 2.7 or higher
* Python pip (or pip3 for Python 3)

We recommend using Python 3 (and pip3) or above. A system with both Python 3 and Python 2.7 may encounter installation error. In that case, please make Python 3 your default choice by appropriate aliasing.

## Install
First install the necessary packages by

`sudo make setup`

then install SLSpy by typing

`sudo make install`

which has been tested under Fedora 30 and Ubuntu 18.04.

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