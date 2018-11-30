# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd

import time
import Re_ranking_load_data as Re_ranking_load_data
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def re_ranking_train(X_feature_action, X_feature_start, X_feature_end, Y_iou, LR, config):
    X = tf.concat((X_feature_start, X_feature_action, X_feature_end), axis=1)
    X = tf.nn.l2_normalize(X, dim=1)
    net = tf.layers.dense(X, units=512, activation=tf.nn.relu,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          bias_initializer=tf.truncated_normal_initializer(stddev=0.01))
    net = tf.nn.dropout(net, 1.0)
    net = tf.layers.dense(net, units=3, activation=None,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          bias_initializer=tf.truncated_normal_initializer(stddev=0.01))

    net_iou = net[:, 0]
    net_offset_pred = net[:, 1:]
    net_iou = tf.nn.sigmoid(net_iou)
    anchors_iou = tf.reshape(net_iou, [-1])
    return anchors_iou, net_offset_pred


class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """

    def __init__(self):
        # common information
        self.training_epochs = 61

        self.input_steps = 256
        self.learning_rates = [0.001] * 15 + [0.0001] * 10

        self.num_random = 10
        self.batch_size = 16
        self.u_ratio_m = 1
        self.u_ratio_l = 2
        self.unit_size = 6.0


if __name__ == "__main__":
    config = Config()
    epoch_idx = 18
    X_feature_action = tf.placeholder(tf.float32, [None, 1024])
    X_feature_start = tf.placeholder(tf.float32, [None, 1024])
    X_feature_end = tf.placeholder(tf.float32, [None, 1024])
    Y_iou = tf.placeholder(tf.float32, [None])
    LR = tf.placeholder(tf.float32)
    prop_score, net_offset_pred = re_ranking_train(X_feature_action, X_feature_start, X_feature_end,
                                                   Y_iou, LR, config)

    model_saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=80)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False
    sess = tf.InteractiveSession(config=tf_config)
    tf.global_variables_initializer().run()
    model_saver.restore(sess, "./models/re_ranking_model_epoch-" + str(epoch_idx))

    dataSet = "Test"
    videoDataTest = Re_ranking_load_data.getFullData(dataSet, flag_test=True)
    annoDf = pd.read_csv("./data/thumos_14_annotations/" + dataSet + "_Annotation.csv")
    videoNameList = list(set(annoDf.video.values[:]))

    for videoName in videoNameList:
        batch_feature_action = videoDataTest[videoName]["feature_action"]
        batch_feature_start = videoDataTest[videoName]["feature_left"]
        batch_feature_end = videoDataTest[videoName]["feature_right"]

        out_score, out_offset = sess.run([prop_score, net_offset_pred],
                                         feed_dict={X_feature_action: batch_feature_action,
                                                    X_feature_start: batch_feature_start,
                                                    X_feature_end: batch_feature_end})

        out_score = np.reshape(out_score, [-1])
        xmin_offset = out_offset[:, 0]
        xmax_offset = out_offset[:, 1]
        xmin_list = videoDataTest[videoName]["xmin"]
        xmax_list = videoDataTest[videoName]["xmax"]
        reg_xmin_list = xmin_list + xmin_offset * config.unit_size
        reg_xmax_list = xmax_list + xmax_offset * config.unit_size

        xmin_score_list = videoDataTest[videoName]["xmin_score"]
        xmax_score_list = videoDataTest[videoName]["xmax_score"]
        latentDf = pd.DataFrame()
        latentDf["xmin"] = xmin_list
        latentDf["xmax"] = xmax_list
        latentDf["xmin_score"] = xmin_score_list
        latentDf["xmax_score"] = xmax_score_list
        latentDf["iou_score"] = out_score

        latentDf["reg_xmin"] = reg_xmin_list
        latentDf["reg_xmax"] = reg_xmax_list

        latentDf.to_csv(
            "./outputs/re_ranking_results_epoch" + str(epoch_idx) + "/" + videoName + ".csv",
            index=False)
