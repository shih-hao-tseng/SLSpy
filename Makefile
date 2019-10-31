install:
	dnf install blas blas-devel
	dnf install lapack lapack-devel
	pip install --no-cache-dir -I scs
	pip install matplotlib
