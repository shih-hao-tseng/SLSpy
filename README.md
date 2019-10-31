python 2.7 or higher
to use the package under python 3, cvxpy for python 3 should be installed manually. the current pip install does not seem to work



install the necessary packages
sudo make install



current Makefile works for Fedora. Ubuntu left to test

might need blas and lapack
sudo dnf install blas blas-devel
sudo dnf install lapack lapack-devel

then reinstall scs
pip install --no-cache-dir --ignore-installed scs
or
pip install --no-cache-dir -I scs
