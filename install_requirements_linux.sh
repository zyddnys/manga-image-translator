sed "/cython/d" < requirements.txt > requirements_linux.txt
pip install -r requirements_linux.txt
