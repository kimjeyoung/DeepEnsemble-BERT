# :fire: DeepEnsemble-BERT :fire:
Implementation of the paper : Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles 

The codes are based on [huggingface](https://huggingface.co/).

Original Paper : [Link](https://papers.nips.cc/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf)

## Installation :coffee:

Training environment : Ubuntu 18.04, python 3.6
```bash
pip3 install torch torchvision torchaudio
pip install scikit-learn
```

Download `bert-base-uncased` checkpoint from [hugginface-ckpt](https://huggingface.co/bert-base-uncased/tree/main)  
Download `bert-base-uncased` vocab file from [hugginface-vocab](https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt)  
Download CLINC OOS intent detection benchmark dataset from [tensorflow-dataset](https://github.com/jereliu/datasets/raw/master/clinc_oos.zip)

The downloaded files' directory should be:

```bash
DeepEnsemble-BERT
ㄴckpt
  ㄴbert-base-uncased-pytorch_model.bin
ㄴdataset
  ㄴclinc_oos
    ㄴtrain.csv
    ㄴval.csv
    ㄴtest.csv
    ㄴtest_ood.csv
  ㄴvocab
    ㄴbert-base-uncased-vocab.txt
ㄴmodels
...
```


## Dataset Info :book:

In their paper, the authors conducted OOD experiment for NLP using CLINC OOS intent detection benchmark dataset, the OOS dataset contains data for 150 in-domain services with 150 training
sentences in each domain, and also 1500 natural out-of-domain utterances.
You can download the dataset at [Link](https://github.com/jereliu/datasets/raw/master/clinc_oos.zip).

Original dataset paper, and Github : [Paper Link](https://aclanthology.org/D19-1131/), [Git Link](https://github.com/clinc/oos-eval)

## Run :star2:

#### Train
```bash
python main.py --train_or_test train --task classification --device gpu --gpu 0
```

#### Test

```bash
python main.py --train_or_test test --task classification --device gpu --gpu 0
```


## References

[1] https://github.com/Kyushik/Predictive-Uncertainty-Estimation-using-Deep-Ensemble  
[2] https://huggingface.co/  
