# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 10:49:40 2017

@author: wzmsltw
"""
import pandas as pd
import numpy as np
import math
import copy
import random
import glob
import os


def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute jaccard score between a box and the anchors.
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    # print inter_len,union_len
    jaccard = np.divide(inter_len, union_len)
    return jaccard


def ioa_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute intersection between score a box and the anchors.
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    scores = np.divide(inter_len, len_anchors)
    return scores


def getBatchList(numWindow, batch_size, shuffle=True):
    ## notice that there are some video appear twice in last two batch ##
    window_list = range(numWindow)
    batch_start_list = [i * batch_size for i in range(len(window_list) / batch_size)]
    batch_start_list.append(len(window_list) - batch_size)
    if shuffle == True:
        random.shuffle(window_list)
    batch_window_list = []
    for bstart in batch_start_list:
        batch_window_list.append(window_list[bstart:(bstart + batch_size)])
    return batch_window_list


def getVideoFeature(videoname, subset):
    appearance_path = '/feature/THUMOS_14_two_stream_feature/{}_appearance/'.format(subset)
    denseflow_path = '/feature/THUMOS_14_two_stream_feature/{}_denseflow/'.format(subset)
    rgb_file_list = glob.glob(appearance_path + videoname + '*')
    flow_file_list = glob.glob(denseflow_path + videoname + '*')

    swin_start = 1.0
    window_size = 6.0
    rgb_feature = []
    for ii in range(len(rgb_file_list)):
        swin_end = swin_start + window_size
        if os.path.exists(appearance_path + videoname + ".mp4" + "_" + str(swin_start) + "_" + str(swin_end) + ".npy"):
            rgb_feat = np.load(
                appearance_path + videoname + ".mp4" + "_" + str(swin_start) + "_" + str(swin_end) + ".npy")
            rgb_feature.append(rgb_feat)
        swin_start = swin_end

    swin_start = 1.0
    flow_feature = []
    for ii in range(len(flow_file_list)):
        swin_end = swin_start + window_size
        if os.path.exists(denseflow_path + videoname + ".mp4" + "_" + str(swin_start) + "_" + str(swin_end) + ".npy"):
            flow_feat = np.load(
                denseflow_path + videoname + ".mp4" + "_" + str(swin_start) + "_" + str(swin_end) + ".npy")
            flow_feature.append(flow_feat)
        swin_start = swin_end

    return rgb_feature, flow_feature


def getBatchData(window_list, data_dict):
    batch_bbox = []
    batch_index = [0]
    batch_anchor_xmin = []
    batch_anchor_xmax = []
    batch_anchor_feature = []
    for idx in window_list:
        batch_bbox.extend(data_dict["gt_bbox"][idx])
        batch_index.append(batch_index[-1] + len(data_dict["gt_bbox"][idx]))
        batch_anchor_xmin.append(data_dict["anchor_xmin"][idx])
        batch_anchor_xmax.append(data_dict["anchor_xmax"][idx])
        batch_anchor_feature.append(data_dict["feature"][idx])
    batch_index = np.array(batch_index)
    batch_bbox = np.array(batch_bbox)
    batch_anchor_xmin = np.array(batch_anchor_xmin)
    batch_anchor_xmax = np.array(batch_anchor_xmax)
    batch_anchor_feature = np.array(batch_anchor_feature)
    # batch_anchor_feature=np.reshape(batch_anchor_feature,[len(video_list),100,-1])
    return batch_index, batch_bbox, batch_anchor_xmin, batch_anchor_xmax, batch_anchor_feature


def getFullData(dataSet):
    ii = 0
    # dataSet="Test"
    annoDf = pd.read_csv("./data/thumos_14_annotations/" + dataSet + "_Annotation.csv")
    videoNameList = list(set(annoDf.video.values[:]))

    list_data = []
    list_anchor_xmins = []
    list_anchor_xmaxs = []
    list_gt_bbox = []

    for videoName in videoNameList:
        print ii
        ii += 1
        video_annoDf = annoDf[annoDf.video == videoName]
        gt_xmins = video_annoDf.startFrame.values[:]
        gt_xmaxs = video_annoDf.endFrame.values[:]
        rgb_feature, flow_feature = getVideoFeature(videoName, dataSet.lower())
        numSnippet = min(len(rgb_feature), len(flow_feature))
        frameList = [1 + 6 * i for i in range(numSnippet)]
        df_data = np.concatenate((rgb_feature, flow_feature), axis=1)
        df_snippet = frameList
        window_size = 100
        stride = window_size / 2
        n_window = (numSnippet + stride - window_size) / stride
        windows_start = [i * stride for i in range(n_window)]
        # print np.shape(df_data)
        if numSnippet < window_size:
            windows_start = [0]
            tmp_data = np.zeros((window_size - numSnippet, 4096))
            df_data = np.concatenate((df_data, tmp_data), axis=0)
            df_snippet.extend([df_snippet[-1] + 6 * (i + 1) for i in range(window_size - numSnippet)])
        elif numSnippet - windows_start[-1] - window_size > 20:
            windows_start.append(numSnippet - window_size)

        snippet_xmin = df_snippet
        snippet_xmax = df_snippet[1:]
        snippet_xmax.append(df_snippet[-1] + 6)
        for start in windows_start:
            tmp_data = df_data[start:start + window_size, :]
            tmp_snippets = np.array(df_snippet[start:start + window_size])
            tmp_anchor_xmins = snippet_xmin[start:start + window_size]
            tmp_anchor_xmaxs = snippet_xmax[start:start + window_size]
            tmp_gt_bbox = []
            tmp_ioa_list = []
            for idx in range(len(gt_xmins)):
                tmp_ioa = ioa_with_anchors(gt_xmins[idx], gt_xmaxs[idx], tmp_anchor_xmins[0], tmp_anchor_xmaxs[-1])
                tmp_ioa_list.append(tmp_ioa)
                if tmp_ioa > 0:
                    tmp_gt_bbox.append([gt_xmins[idx], gt_xmaxs[idx]])
            # print tmp_ioa_list
            if len(tmp_gt_bbox) > 0 and max(tmp_ioa_list) > 0.9:
                list_gt_bbox.append(tmp_gt_bbox)
                list_anchor_xmins.append(tmp_anchor_xmins)
                list_anchor_xmaxs.append(tmp_anchor_xmaxs)
                list_data.append(tmp_data)
    dataDict = {"gt_bbox": list_gt_bbox, "anchor_xmin": list_anchor_xmins, "anchor_xmax": list_anchor_xmaxs,
                "feature": list_data}
    return dataDict


def getVideoData(videoName, subset):
    list_data = []
    list_snippets = []
    rgb_feature, flow_feature = getVideoFeature(videoName, subset)
    numSnippet = min(len(rgb_feature), len(flow_feature))
    frameList = [1 + 6 * i for i in range(numSnippet)]
    df_data = np.concatenate((rgb_feature, flow_feature), axis=1)
    df_snippet = frameList
    window_size = 100
    stride = window_size / 2
    n_window = (numSnippet + stride - window_size) / stride
    windows_start = [i * stride for i in range(n_window)]

    if numSnippet < window_size:
        windows_start = [0]
        tmp_data = np.zeros((window_size - numSnippet, 4096))
        df_data = np.concatenate((df_data, tmp_data), axis=0)
        df_snippet.extend([df_snippet[-1] + 6 * (i + 1) for i in range(window_size - numSnippet)])
    else:
        windows_start.append(numSnippet - window_size)

    for start in windows_start:
        tmp_data = df_data[start:start + window_size, :]
        tmp_snippets = np.array(df_snippet[start:start + window_size])
        list_data.append(tmp_data)
        list_snippets.append(tmp_snippets)

    list_snippets = np.array(list_snippets)
    list_data = np.array(list_data)
    return list_snippets, list_data, df_snippet
