source ./env_tcc_eeg/Scripts/activate

python src/train/train_cnn.py > .\src\artifacts\train_cnn.txt

python src/train/train_transformer.py > .\src\artifacts\train_transformer.txt

python src/train/train_ml.py > .\src\artifacts\train_ml.txt