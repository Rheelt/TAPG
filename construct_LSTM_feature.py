# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
import TAPG_load_data as TAPG_load_data
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def tapg_inference(X_feature, config):
    layer1 = tf.layers.conv1d(inputs=X_feature, filters=512, kernel_size=3, strides=1, padding='same',
                              activation=tf.nn.relu)
    layer2 = tf.layers.conv1d(inputs=layer1, filters=512, kernel_size=3, strides=1, padding='same', activation=None)
    layer3 = tf.add(layer1, layer2)
    net = tf.nn.relu(layer3)
    with tf.variable_scope(name_or_scope='init', initializer=tf.orthogonal_initializer()):
        fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=512, state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=512, state_is_tuple=True)
        fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=1.0)
        bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=1.0)
        output_rnn, final_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, net, dtype=tf.float32)
    concat_output_rnn = tf.concat(output_rnn, axis=-1)
    net = 0.1 * tf.layers.conv1d(inputs=concat_output_rnn, filters=3, kernel_size=1, strides=1,
                                 padding='same')
    scores = tf.nn.sigmoid(net)

    trainable_variables = tf.trainable_variables()
    return concat_output_rnn, scores, trainable_variables


class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """

    def __init__(self):
        # common information
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
    epoch_idx = 8
    fetch_concat_output_rnn, scores, trainable_variables = tapg_inference(X_feature, config)
    model_saver = tf.train.Saver(var_list=trainable_variables, max_to_keep=80)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False
    sess = tf.InteractiveSession(config=tf_config)
    tf.global_variables_initializer().run()
    model_saver.restore(sess, "models/tapg_model_epoch-" + str(epoch_idx))

    annoDf_train = pd.read_csv("./data/thumos_14_annotations/Val_Annotation.csv")
    annoDf_test = pd.read_csv("./data/thumos_14_annotations/Test_Annotation.csv")
    videoNameList = list(set(annoDf_train.video.values[:])) + list(set(annoDf_test.video.values[:]))
    valNameList = set(annoDf_train.video.values[:])
    testNameList = set(annoDf_test.video.values[:])
    columns = ["frame", "action", "start", "end"]
    concat_output_rnn_dim = 1024

    columns_rnn = ["frame"]
    for i in range(concat_output_rnn_dim):
        columns_rnn.append("f" + str(i))
    for videoName in videoNameList:
        if videoName in valNameList:
            subset = 'val'
        elif videoName in testNameList:
            subset = 'test'
        list_snippets, list_data, video_snippet = TAPG_load_data.getVideoData(videoName, subset)
        concat_output_rnn, out_scores = sess.run([fetch_concat_output_rnn, scores], feed_dict={X_feature: list_data})
        calc_time_list = np.zeros(len(video_snippet))
        snippet_scores = np.zeros([len(video_snippet), 3])
        snippet_concat_output_rnn = np.zeros([len(video_snippet), concat_output_rnn_dim])

        for idx in range(len(list_snippets)):
            snippets = list_snippets[idx]
            for jdx in range(len(snippets)):
                tmp_snippet = snippets[jdx]
                tmp_snippet_index = video_snippet.index(tmp_snippet)
                calc_time_list[tmp_snippet_index] += 1
                snippet_scores[tmp_snippet_index, :] += out_scores[idx, jdx, :]
                snippet_concat_output_rnn[tmp_snippet_index, :] += concat_output_rnn[idx, jdx, :]

        calc_time_list = np.reshape(calc_time_list, [-1, 1])
        snippet_scores = snippet_scores / calc_time_list
        snippet_concat_output_rnn = snippet_concat_output_rnn / calc_time_list

        snippet_scores = np.concatenate((np.reshape(video_snippet, [-1, 1]), snippet_scores), axis=1)
        snippet_concat_output_rnn = np.concatenate((np.reshape(video_snippet, [-1, 1]), snippet_concat_output_rnn),
                                                   axis=1)

        tmp_df = pd.DataFrame(snippet_scores, columns=columns, dtype=np.float32)
        rnn_tmp_df = pd.DataFrame(snippet_concat_output_rnn, columns=columns_rnn, dtype=np.float32)
        tmp_df.to_csv(
            "output/TAPG_results_epoch" + str(epoch_idx) + "/" + videoName + ".csv",
            index=False)
        rnn_tmp_df.to_csv(
            "output/TAPG_results_epoch" + str(
                epoch_idx) + "/" + videoName + "_rnn.csv",
            index=False)
