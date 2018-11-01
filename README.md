# Deep Learning for Text Pairs Relation Classification

This repository is my bachelor graduation project, and it is also a study of TensorFlow, Deep Learning(CNN, RNN, LSTM, etc.).

The main objective of the project is to determine whether the two sentences are similar in sentence meaning (binary classification problems) by the two given sentences based on Neural Networks (Fasttext, CNN, LSTM, etc.).

## Requirements

- Python 3.6
- Tensorflow 1.8 +
- Numpy
- Gensim

## Innovation

### Data part
1. Make the data support **Chinese** and English.(Which use `jieba` seems easy)
2. Can use **your own pre-trained word vectors**.(Which use `gensim` seems easy)
3. Add embedding visualization based on the **tensorboard**.

### Model part
1. Deign **two subnetworks** to solve the task --- Text Pairs Similarity Classification.
2. Add the correct **L2 loss** calculation operation.
3. Add **gradients clip** operation to prevent gradient explosion.
4. Add **learning rate decay** with exponential decay.
5. Add a new **Highway Layer**.(Which is useful according to the model performance)
6. Add **Batch Normalization Layer**.
7. Add several performance measures(especially the **AUC**) since the data is imbalanced.

### Code part
1. Can choose to **train** the model directly or **restore** the model from checkpoint in `train.py`.
2. Add `test.py`, the **model test code**, it can show the predict value of label of the data in Testset when create the final prediction file.
3. Add other useful data preprocess functions in `data_helpers.py`.
4. Use `logging` for helping recording the whole info(including parameters display, model training info, etc.).
5. Provide the ability to save the best n checkpoints in `checkmate.py`, whereas the `tf.train.Saver` can only save the last n checkpoints.

## Data

See data format in `data` folder which including the data sample files.

### Text Segment

You can use `jieba` package if you are going to deal with the chinese text data.

### Data Format

This repository can be used in other datasets(text pairs similarity classification) by two ways:
1. Modify your datasets into the same format of the sample.
2. Modify the data preprocess code in `data_helpers.py`.

Anyway, it should depends on what your data and task are.

### Pre-trained Word Vectors

You can pre-training your word vectors(based on your corpus) in many ways:
- Use `gensim` package to pre-train data.
- Use `glove` tools to pre-train data.
- Even can use a **fasttext** network to pre-train data.

## Network Structure

### FastText

![](https://farm2.staticflickr.com/1908/43814433590_4ba52aec62_o.png)

References:

- [Bag of Tricks for Efficient Text Classification](https://arxiv.org/pdf/1607.01759.pdf)

---

### TextANN

![](https://farm2.staticflickr.com/1960/43841530260_9b69219791_o.png)

References:

- **Personal ideas 🙃**

---


### TextCNN

![](https://farm2.staticflickr.com/1979/44907317574_cd090e115f_o.png)

References:

- [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)
- [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1510.03820)

---

### TextRNN

**Warning: Model can use but not finished yet 🤪!**

![](https://farm2.staticflickr.com/1934/44745166925_4b2e4928e3_o.png)

#### TODO

1. Add BN-LSTM cell unit.
2. Add attention.

References:

- [Recurrent Neural Network for Text Classification with Multi-Task Learning](http://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552)

---

### TextCRNN

![](https://farm2.staticflickr.com/1965/43815348930_b9e0824cb6_o.png)

References:

- **Personal ideas 🙃**

---

### TextRCNN

References:

- **Personal ideas 🙃**

---

### TextHAN

References:

- [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)

---

### TextSANN

**Warning: Model can use but not finished yet 🤪!**

#### TODO
1. Add attention penalization loss.
2. Add visualization.

References:

- [A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING](https://arxiv.org/pdf/1703.03130.pdf)

---

### TextABCNN

**Warning: Only achieve the ABCNN-1 Model🤪!**

#### TODO

1. Add ABCNN-3 model.

References:

- [ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs](https://arxiv.org/pdf/1512.05193.pdf)

---

## About Me

黄威，Randolph

SCU SE Bachelor; USTC CS Master

Email: chinawolfman@hotmail.com

My Blog: [randolph.pro](http://randolph.pro)

LinkedIn: [randolph's linkedin](https://www.linkedin.com/in/randolph-%E9%BB%84%E5%A8%81/)
