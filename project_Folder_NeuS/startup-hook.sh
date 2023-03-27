apt-get update && apt-get install -y wget
cd NeuS
pip install -r requirements.txt
apt-get install libgl1 -y
cd ..
mv NeuS code/
cd code/NeuS/
