Fake News Classification (Data Science II Project)
=================================================

Fake news classification using deep learning (CNN+LSTM with BERT embeddings and fine-tuned BERT).

This project investigates the effectiveness of two deep learning approaches for fake news
classification:

1) CNN + LSTM trained on frozen BERT embeddings
2) Fine-tuned BERT classifier (end-to-end)

Datasets
--------

- ISOT Fake News Dataset: True.csv, Fake.csv
- Kaggle / WELFake Dataset: WELFake_Dataset.csv

Dataset CSV files are not included in this repository. Download them separately and place
them in the project directory.

Project Structure
-----------------

ai-data-science-fake-news-detection-bert/
  src/
  outputs/
  requirements.txt
  README.md
  LICENSE

Setup
-----

Install dependencies:

pip install -r requirements.txt

Usage
-----

1) Preprocess (creates train/val/test splits)

python -m src.preprocess --isot_true True.csv --isot_fake Fake.csv --kaggle WELFake_Dataset.csv --out_dir outputs

2) Train Model 1 (CNN + LSTM on BERT embeddings)

python -m src.train_cnn_lstm --dataset isot --data_dir outputs --epochs 3
python -m src.train_cnn_lstm --dataset kaggle --data_dir outputs --epochs 3

3) Train Model 2 (Fine-tuned BERT)

python -m src.train_bert --dataset isot --data_dir outputs --epochs 2
python -m src.train_bert --dataset kaggle --data_dir outputs --epochs 2

Results
-------

Test performance on both datasets:

Model                         Dataset   Accuracy   Precision   Recall    F1
CNN + LSTM (BERT embeddings)   ISOT      0.9994     0.9989      1.0000    0.9994
CNN + LSTM (BERT embeddings)   Kaggle    0.9901     0.9946      0.9862    0.9903
Fine-tuned BERT                ISOT      0.9993     0.9989      0.9997    0.9993
Fine-tuned BERT                Kaggle    0.9940     0.9930      0.9953    0.9942

Outputs
-------

- outputs/results_cnn_lstm_{dataset}.json
- outputs/results_bert_{dataset}.json

Author
------

Yazan Aqtash  
GitHub: https://github.com/Yazanoov-eng

License
-------

MIT License

