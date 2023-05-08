apt-get update && apt-get install -y wget
apt-get install libgl1 -y
# # pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 torchtext==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
export TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
pip install torch-scatter -f https://data.pyg.org/whl/torch-$TORCH_VERSION.html
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install pycollada
# cd /home/battocchiozinni/ngp_pl/apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ngp_pl && pip install -r requirements.txt
pip install models/csrc/
cd ..
mv ngp_pl code/
# apt-get install python3.8-venv -y
# python -m venv batzin_venv
# source batzin_venv/bin/activate
# pip install determined
# pip install pycollada