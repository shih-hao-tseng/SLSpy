setup:
	dnf install blas blas-devel
	dnf install lapack lapack-devel
	pip install --no-cache-dir -I scs
	pip install matplotlib

install:
	pip install -e slspy