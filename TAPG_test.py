# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 20:25:55 2017

@author: wzmsltw
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import TEM_load_data_LSTM as TEM_load_data
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"


def tem_inference(X_feature, config):

    layer1 = tf.layers.conv1d(inputs=X_feature, filters=512, kernel_size=3, strides=1, padding='same',
                                  activation=tf.nn.relu)
    layer2 = tf.layers.conv1d(inputs=layer1, filters=512, kernel_size=3, strides=1, padding='same', activation=None)
    layer3 = tf.add(layer1, layer2)
    net = tf.nn.relu(layer3)
    with tf.variable_scope(name_or_scope='init', initializer=tf.orthogonal_initializer()):
        fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=512, state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=512, state_is_tuple=True)
        fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=1)
        bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=1)
        output_rnn, final_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, net, dtype=tf.float32)
    concat_output_rnn = tf.concat(output_rnn, axis=-1)
    net = 0.1 * tf.layers.conv1d(inputs=concat_output_rnn, filters=3, kernel_size=1, strides=1,
                                     padding='same')
    scores = tf.nn.sigmoid(net)
    TEM_trainable_variables = tf.trainable_variables()
    return scores, TEM_trainable_variables


class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """

    def __init__(self):
        # common information
        self.learning_rates = [0.001] * 5 + [0.0001] * 15
        self.training_epochs = len(self.learning_rates)
        self.n_inputs = 4096
        self.negative_ratio = 1
        self.batch_size = 16
        self.num_prop = 100


if __name__ == "__main__":
    config = Config()
    X_feature = tf.placeholder(tf.float32, shape=(None, config.num_prop, config.n_inputs))
    Y_bbox = tf.placeholder(tf.float32, [None, 2])
    Index = tf.placeholder(tf.int32, [config.batch_size + 1])
    LR = tf.placeholder(tf.float32)
    epoch_idx = 11
    scores, TEM_trainable_variables = tem_inference(X_feature, config)

    model_saver = tf.train.Saver(var_list=TEM_trainable_variables, max_to_keep=80)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False
    sess = tf.InteractiveSession(config=tf_config)
    tf.global_variables_initializer().run()
    # model_saver.restore(sess, "models/TEM/tem_model_epoch-7")
    model_saver.restore(sess, "./ablation_study/hy_lambda_4_6/TEM/tem_model_epoch-" + str(epoch_idx))

    annoDf_train = pd.read_csv("./data/thumos_14_annotations/Val_Annotation.csv")
    annoDf_test = pd.read_csv("./data/thumos_14_annotations/Test_Annotation.csv")
    videoNameList = list(set(annoDf_train.video.values[:])) + list(set(annoDf_test.video.values[:]))
    valNameList = set(annoDf_train.video.values[:])
    testNameList = set(annoDf_test.video.values[:])
    columns = ["frame", "action", "start", "end"]

    for videoName in videoNameList:
        if videoName in valNameList:
            subset = 'val'
        elif videoName in testNameList:
            subset = 'test'
        list_snippets, list_data, video_snippet = TEM_load_data.getVideoData(videoName, subset)
        out_scores = sess.run(scores, feed_dict={X_feature: list_data})
        calc_time_list = np.zeros(len(video_snippet))
        snippet_scores = np.zeros([len(video_snippet), 3])

        for idx in range(len(list_snippets)):
            snippets = list_snippets[idx]
            for jdx in range(len(snippets)):
                tmp_snippet = snippets[jdx]
                tmp_snippet_index = video_snippet.index(tmp_snippet)
                calc_time_list[tmp_snippet_index] += 1
                snippet_scores[tmp_snippet_index, :] += out_scores[idx, jdx, :]

        calc_time_list = np.stack([calc_time_list, calc_time_list, calc_time_list], axis=1)
        snippet_scores = snippet_scores / calc_time_list

        snippet_scores = np.concatenate((np.reshape(video_snippet, [-1, 1]), snippet_scores), axis=1)
        tmp_df = pd.DataFrame(snippet_scores, columns=columns)
        tmp_df.to_csv(
            "output/TEM_results_epoch" + str(epoch_idx) + "/" + videoName + ".csv",
            index=False)
