import numpy as np
import cv2
import os
import pickle


def getCenterCoord(box):
    return np.array([int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)])


def getBoundingImg(frame, box, crop_fr=(0.1, 0.2), size=(128, 192)):
    x1, y1, x2, y2 = box[:4].astype(int)
    width = x2 - x1
    height = y2 - y1
    x_min = max(0, x1 - crop_fr[0] * width)
    x_max = min(x2 + crop_fr[0] * width, frame.shape[1])
    y_min = max(0, y1 - crop_fr[1] * height)
    y_max = min(y2 + crop_fr[1] / 3 * height, frame.shape[0])
    return cv2.resize(frame[int(y_min):int(y_max), int(x_min):int(x_max), :], size)


def similarity_matrix(l1, l2):
    N = len(l1)
    M = len(l2)
    res = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            res[i, j] = np.dot(l1[i], l2[j])

    return res


def get_name_from_base(feat_vec, feat_base):
    name = 'Undefined'
    max_sim = 0.5
    kpp_base, short_term_base = feat_base
    for key in kpp_base.keys():
        sim = np.dot(feat_vec, kpp_base[key][0])
        if sim > max_sim:
            max_sim = sim
            name = key
    # tracking known face
    if max_sim > 0.5: return name

    for key in short_term_base.keys():
        sim = np.dot(feat_vec, short_term_base[key])
        if sim > max_sim:
            max_sim = sim
            name = key

    return name


def create_base_from_imgs(path, model):
    print("Collecting database...")
    features_base = {}
    id = 0
    for filename in os.listdir(path):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(path, filename))
            fimgs, _, _, _ = model.detect_faces(img)
            # print(filename,criterions[0])
            features_base[id] = model.get_feature(fimgs[0])
            id += 1

    return features_base


def get_base_from_file(path):
    return pickle.load(open(path, "rb"))
