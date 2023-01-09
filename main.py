from dagmm import DAGMM
import tensorflow as tf
import pandas as pd

train_path = 'XXXXXX/train_data.csv'
test_path = 'XXXXXX/test_data.csv'
x = pd.read_csv(train_path, encoding='gb18030').values
y = pd.read_csv(test_path, encoding='gb18030').values

model = DAGMM(
    comp_hiddens=[8, 4, 1], comp_activation=tf.nn.tanh,
    est_hiddens=[8, 8], est_activation=tf.nn.tanh, est_dropout_ratio=0.1,
    epoch_size=50, minibatch_size=256
)

model.fit(x)   # model training
model.predict(y)   # sample energy

