# -*- coding: utf-8 -*-
import tensorflow as tf

class EstimationNet:
    """ Estimation Network

    This network converts input feature vector to softmax probability.
    Bacause loss function for this network is not defined,
    it should be implemented outside of this class.
    """
    def __init__(self, hidden_layer_sizes, activation=tf.nn.relu):

        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation

    def inference(self, z, dropout_ratio=None):

        with tf.variable_scope("EstNet"):
            n_layer = 0
            for size in self.hidden_layer_sizes[:-1]:
                n_layer += 1
                z = tf.layers.dense(z, size, activation=self.activation,
                    name="layer_{}".format(n_layer))
                if dropout_ratio is not None:
                    z = tf.layers.dropout(z, dropout_ratio,
                        name="drop_{}".format(n_layer))

            size = self.hidden_layer_sizes[-1]
            logits = tf.layers.dense(z, size, activation=None, name="logits")

            output = tf.contrib.layers.softmax(logits)

        return output
