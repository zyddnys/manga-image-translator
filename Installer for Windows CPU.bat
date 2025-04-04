@rem The disk where the anaconda is installed
@rem Change the path to the activate.bat file
F:
call F:\1\Scripts\activate.bat
conda create -n manga-image-translator python=3.10 -y
conda activate manga-image-translator
pip3 install torch torchvision torchaudio
pip install -r requirements.txt
conda install -c conda-forge pydensecrf -y
