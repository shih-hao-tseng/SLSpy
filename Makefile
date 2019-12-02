setup:
	dnf install python3-devel
	dnf install blas blas-devel
	dnf install lapack lapack-devel
	pip install --no-cache-dir -I scs
	pip install cvxpy
	pip install matplotlib

install:
	pip install -e .

uninstall:
	pip uninstall slspy
