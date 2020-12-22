# Important frame in sports motion detector

## Requirement

```bash
conda create -n mildetector python=3.5
conda activate mildetector
conda install -y ffmpeg 
conda install -y opencv=3.4.1 numpy=1.15.1 scipy scikit-learn cvxopt matplotlib -c conda-forge

# easy version
# pip install -e git+https://github.com/garydoranjr/misvm.git#egg=misvm
git clone https://github.com/garydoranjr/misvm.git
cd misvm
python setup.py install
```