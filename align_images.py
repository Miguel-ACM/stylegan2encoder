import os
import sys
import bz2
import json
from keras.utils import get_file
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector
import pickle
LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
DIRS_DATABASE = ['/home/miguel/Desktop/TFM/faces_py/datasets/alignments/data/wflw/train.json']#,
                #'/home/miguel/Desktop/TFM/faces_py/datasets/alignments/data/wflw/valid.json',
                #'/home/miguel/Desktop/TFM/faces_py/datasets/alignments/data/wflw/test.json']

jsons = []
for file in DIRS_DATABASE:
    f = open(file, 'r')
    jsons.append(json.load(f))


def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path


def find_entries(filename, file_format='jpg'):
    to_search = filename + "." + file_format if file_format is not None else filename
    res = []
    for data in jsons:
        for entry in data:
            if to_search in entry['imgpath']:
                res.append(entry)

    return res


def overlapping_bbox(bbox1, bbox2):
    #If one rectangle is on left side of other
    if ((bbox1[0] >= bbox2[2]) or (bbox2[0] >= (bbox1[0] + bbox1[2]))):
        return False

    #If one rectangle is above other
    if ((bbox1[1] >= bbox2[3]) or (bbox2[1] >= (bbox1[1] + bbox1[3]))):
        return False

    dx = min(bbox1[0] + bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0])
    dy = min(bbox1[1] + bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1])
    d = dx * dy
    if d < ((bbox1[2]) * (bbox1[3]) * 0.75):
        return False

    return True


def bbox_size(bbox1):
    return bbox1[2] * bbox1[3]

MIN_NUM_PIXELS_FACE = 22500 #150 * 150
def explore_dirs(path, extra_path, crops, recursive=True):
    aligned_dir = os.path.join(ALIGNED_IMAGES_DIR, extra_path)
    for img_name in [f for f in os.listdir(path)]:
        raw_img_path = os.path.join(path, img_name)
        if os.path.isdir(raw_img_path):
            if recursive:
                crops = explore_dirs(os.path.join(path, img_name), os.path.join(extra_path, img_name), crops)
        else:
            entries = find_entries(img_name, None)
            for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
                face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
                aligned_face_path = os.path.join(aligned_dir, face_img_name)
                crop, img = image_align(raw_img_path, aligned_face_path, face_landmarks)
                # Save aligned image.
                if len(entries) > 0:
                    for entry in entries:
                        if bbox_size(entry['bbox']) > MIN_NUM_PIXELS_FACE and overlapping_bbox(entry['bbox'], crop['crop']):
                            os.makedirs(aligned_dir, exist_ok=True)
                            crop["entry"] = entry
                            crops[face_img_name.split(".")[0]] = crop
                            # Save aligned image.
                            img.save(aligned_face_path, 'PNG')
                            break


    return crops

if __name__ == "__main__":
    """
    Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
    python align_images.py /raw_images /aligned_images
    """

    landmarks_model_path = unpack_bz2(get_file('shape_predictor_68_face_landmarks.dat.bz2',
                                               LANDMARKS_MODEL_URL, cache_subdir='temp'))
    RAW_IMAGES_DIR = sys.argv[1]
    ALIGNED_IMAGES_DIR = sys.argv[2]

    try:
        f = open(os.path.join(ALIGNED_IMAGES_DIR, "crops.pickle"), "rb")
        crops = pickle.load(f)
        f.close()
    except IOError:
        crops = {}

    landmarks_detector = LandmarksDetector(landmarks_model_path)

    crops = explore_dirs(RAW_IMAGES_DIR, '', crops, True)

    pickle.dump(crops, open(os.path.join(ALIGNED_IMAGES_DIR, "crops.pickle"), "wb"))
