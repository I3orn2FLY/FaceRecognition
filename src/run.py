import cv2
import numpy as np
import glob
import os
from detection import MtcnnDetector

detector = MtcnnDetector()


def get_name(feat_base, feat):
    name = "Undefined"
    max_sim = 0
    for cand in feat_base.keys():
        sim = feat_base[cand].dot(feat)
        if sim > max_sim:
            max_sim = sim
            if sim > 0.5:
                name = cand

    return name


feat_base = {}
print("Processing Base...")
for file in glob.glob("../LabeledFaces/*"):
    img = cv2.imread(file)
    feats, boxes, _, landmarks_list = detector.detect_faces(img)
    areas = [(box[0] - box[2]) * (box[1] - box[3]) for box in boxes]
    idx = np.argmax(areas)
    feat = feats[idx]
    name = os.path.splitext(os.path.split(file)[1])[0]
    feat_base[name] = feat
    print(name)

print("Base processed.")

img = cv2.imread('test.jpg')

feats, boxes, _, landmarks_list = detector.detect_faces(img)

for feat, box in zip(feats, boxes):
    box = np.int32(box)
    found_name = get_name(feat_base, feat)
    cv2.putText(img, found_name, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1)

cv2.imshow("Window", img)

cv2.imwrite("result.jpg", img)
cv2.waitKey(0)
