# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


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


annoDf_train = pd.read_csv("./data/thumos_14_annotations/Val_Annotation.csv")
annoDf_test = pd.read_csv("./data/thumos_14_annotations/Test_Annotation.csv")
annoDf = pd.concat([annoDf_train, annoDf_test])
videoNameList = list(set(annoDf.video.values[:]))
epoch_idx = 11
for videoName in videoNameList:

    video_annoDf = annoDf[annoDf.video == videoName]
    tapg_df = pd.read_csv(
        "output/TAPG_results_epoch" + str(epoch_idx) + "/" + videoName + ".csv")

    frame_list = tapg_df.frame.values[:]
    start_scores = tapg_df.start.values[:]
    end_scores = tapg_df.end.values[:]

    """Generate candidate starting and ending locations"""
    start_bins = np.zeros(len(start_scores))
    start_bins[[0, -1]] = 1
    for idx in range(1, len(start_scores) - 1):
        if start_scores[idx] > start_scores[idx + 1] and start_scores[idx] > start_scores[idx - 1]:
            if start_scores[idx] > 0.9:
                start_bins[[idx, idx - 1, idx + 1]] = 1
            else:
                start_bins[idx] = 1

    end_bins = np.zeros(len(end_scores))
    end_bins[[0, -1]] = 1
    for idx in range(1, len(start_scores) - 1):
        if end_scores[idx] > end_scores[idx + 1] and end_scores[idx] > end_scores[idx - 1]:
            if end_scores[idx] > 0.9:
                end_bins[[idx, idx - 1, idx + 1]] = 1
            else:
                end_bins[idx] = 1

    xmin_list = []
    xmin_score_list = []
    xmax_list = []
    xmax_score_list = []
    for j in range(len(start_scores)):
        if start_bins[j] == 1:
            xmin_list.append(frame_list[j])
            xmin_score_list.append(start_scores[j])
        if end_bins[j] == 1:
            xmax_list.append(frame_list[j])
            xmax_score_list.append(end_scores[j])

    """Generate candidate proposals"""
    new_props = []
    for ii in range(len(xmax_list)):
        tmp_xmax = xmax_list[ii]
        tmp_xmax_score = xmax_score_list[ii]

        for ij in range(len(xmin_list)):
            tmp_xmin = xmin_list[ij]
            tmp_xmin_score = xmin_score_list[ij]
            if tmp_xmax - tmp_xmin < 10:
                break
            if tmp_xmax - tmp_xmin > 300:
                continue
            new_props.append([tmp_xmin, tmp_xmax, tmp_xmin_score, tmp_xmax_score])
    new_props = np.stack(new_props)

    col_name = ["xmin", "xmax", "xmin_score", "xmax_score"]
    new_df = pd.DataFrame(new_props, columns=col_name)

    """Calculate intersection of candidate proposals with ground-truth"""
    gt_xmins = video_annoDf.startFrame.values[:]
    gt_xmaxs = video_annoDf.endFrame.values[:]

    new_iou_list = []
    match_xmin_list = []
    match_xmax_list = []
    for j in range(len(new_df)):
        tmp_new_iou = list(iou_with_anchors(new_df.xmin.values[j], new_df.xmax.values[j], gt_xmins, gt_xmaxs))
        new_iou_list.append(max(tmp_new_iou))
        match_xmin_list.append(gt_xmins[tmp_new_iou.index(max(tmp_new_iou))])
        match_xmax_list.append(gt_xmaxs[tmp_new_iou.index(max(tmp_new_iou))])

    new_ioa_list = []
    for j in range(len(new_df)):
        tmp_new_ioa = max(ioa_with_anchors(new_df.xmin.values[j], new_df.xmax.values[j], gt_xmins, gt_xmaxs))
        new_ioa_list.append(tmp_new_ioa)

    new_df["match_iou"] = new_iou_list
    new_df["match_ioa"] = new_ioa_list
    new_df["match_xmin"] = match_xmin_list
    new_df["match_xmax"] = match_xmax_list

    new_df.to_csv("outputs/init_results_epoch" + str(epoch_idx) + "/" + videoName + ".csv",
                  index=False)
