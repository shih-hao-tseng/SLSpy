# SLSpy: Python-based System-Level Controller Synthesis Framework

## Synopsis
*SLSpy* provides a Python-based framework to design and simulate model-based control systems, especially for system level synthesis (SLS) methods. The details of the framework are described in the paper

Shih-Hao Tseng and Jing Shuang (Lisa) Li, ``[SLSpy: Python-Based System-Level Controller Synthesis Framework](https://arxiv.org/abs/2004.12565),'' 2020.

The synthesis workflow follows the structure in the paper

Shih-Hao Tseng and James Anderson, ``[Deployment Architectures for Cyber-Physical Control Systems](https://arxiv.org/abs/1911.01510),'' 2019. (also in *Proc. IEEE ACC*, 2020)

## System Requirement
<!--
* Python 2.7 or higher
* Python pip (or pip3 for Python 3)

We recommend using Python 3 (and pip3) or above. 
-->
* Python 3 or higher
* Python pip3

A system with both Python 3 and Python 2.7 may encounter installation error. In that case, please make Python 3 your default choice by appropriate aliasing.

SLSpy might still work for Python 2.7 (or higher Python 2 versions), but one should ensure CVXPY is version 1.1 or higher.

## Install
First install the necessary packages by

`sudo make setup`

then install SLSpy by typing

`sudo make install`

which has been tested under Fedora 30 and Ubuntu 18.04.

For Windows Subsystem for Linux and macOS, the user might need to install without `sudo` after the setup by

`make install`

<!--
## SLSpy Wiki
A simple wiki page with basic explanation can be found at 
http://slswiki.cms.caltech.edu/index.php/SLSpy

(The server is currently down. We will bring it back when the server is up again.
-->
## Get Started
To get started, please refer to the examples in the folder ``examples.''

For more detailed explanations about the basic framework structure, please refer to 

Shih-Hao Tseng and Jing Shuang (Lisa) Li, ``[SLSpy: Python-Based System-Level Controller Synthesis Framework](https://arxiv.org/abs/2004.12565),'' 2020.

## Built-in Synthesis Algorithms
* System Level Synthesis (FIR version), c.f.\
  James Anderson, John C. Doyle, Steven H. Low, and Nikolai Matni,\
  ``System Level Synthesis,'' *Annual Reviews in Control*, 2019.

  Yuh-Shyang Wang, Nikolai Matni, and John C. Doyle,\
  ``A System Level Approach to Controller Synthesis,'' *IEEE Trans. Autom. Control*, 2019

* Input-Output Parameterization (FIR version), c.f.\
  Luca Furieri, Yang Zheng, Antonis Papachristodoulou, and Maryam Kamgarpour,\
  ``An Input-Output Parameterization of Stabilizing Controllers: Amidst Youla and System Level Synthesis,'' *IEEE Control Systems Letters*, 2019

## Applicable Scenario
Currently, SLSpy can handle a linear time-invariant (LTI) system by synthesizing finite impulse response (FIR) controller using SLS/IOP. It is possible to deal with linear time-variant (LTV) system under SLSpy framework by introducing customized system model, controller model, and synthesizer.

The way SLSpy works is to express transfer matrices as a list. For example, an FIR transfer matrix <b>Φ</b> with the z-transform

<b>Φ</b> = Σ_{t=0}^T Φ[t] z^{-t}

is expressed by

[ Φ[0], Φ[1], ..., Φ[T] ]

in the codes.