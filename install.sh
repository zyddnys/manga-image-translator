sed "/cython/d" < requirements.txt > requirements_linux.txt
sed -i 's|https://www.lfd.uci.edu/~gohlke/pythonlibs/#_pydensecrf|https://github.com/lucasb-eyer/pydensecrf|g' requirements_linux.txt
pip install -r requirements_linux.txt
