# upr-kilt
UPR for retrieval on KILT

### Setup

Choose a base directory (`BASE_DIR`) which will be large enough to hold upto 60 GB of data.

#### Clone

```{bash}
# Clone this repo
git clone git@github.com:VedangW/upr-kilt.git

# Clone KILT
git clone git@github.com:facebookresearch/KILT.git
```

#### Install requirements

```{bash}
cd upr-kilt

conda create -n uprkilt python=3.8
conda activate uprkilt

pip install -r requirements.txt
pip3 install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
conda install -c pytorch faiss-gpu

cd ../KILT
python setup.py install

cd ../upr-kilt
```

#### Install KILT datasets

```{bash}
# Change BASE_DIR in kilt_setup.sh
bash scripts/kilt_setup.sh
```