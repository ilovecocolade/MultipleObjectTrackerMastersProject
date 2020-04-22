import matplotlib
matplotlib.use('Agg')

import cv2
import skimage.io
import glob
import mrcnn.model as modellib
from mrcnn import utils
from mrcnn import visualize
from samples.coco import coco
import os
import datetime
import skimage.draw
from matplotlib import pyplot as plt
import numpy as np
import shutil
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Root directory of the project
ROOT_DIR = os.path.abspath('maskRCNN/')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.abspath("mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
#if not os.path.exists(COCO_MODEL_PATH):
 #5   utils.download_trained_weights(COCO_MODEL_PATH)


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def segment_frames(model, video_path=None, class_names=None, write_path=None, start_frame=0, txt_name=None):

    rep_store = []

    # abstracts only the detections of desired classes
    desired_classes = [class_names.index('person')]

    # Parse Image file
    img_files = sorted(glob.glob(os.path.join(video_path, '*.jp*')))
    ims = [cv2.imread(imf) for imf in img_files]

    txt_file = open(txt_name + ".txt", "w+")

    for count, image in enumerate(ims):
        if count >= start_frame:
            print("frame: ", count)

            image = image[..., ::-1]
            # Detect objects
            r = model.detect([image], verbose=0)[0]
            # Color splash

            r = get_desired_results(r, desired_classes)

            splash, rep_store, colors, boxes = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                                                class_names, count, r['scores'], colors=None,
                                                                making_video=True, instance_segmentation=True,
                                                                score_thresh=0, rep_store=rep_store)

            for x in range(np.shape(colors)[0]):

                entry = str((count, colors[x, 0], colors[x, 1], colors[x, 2], boxes[x, 0], boxes[x, 1],
                             boxes[x, 2] - boxes[x, 0], boxes[x, 3] - boxes[x, 1]))

                txt_file.write(entry + '\r\n')

            cv2.imwrite(os.path.join(write_path, '%05d.jpg' % count), splash)

    txt_file.close()


def segment_video(model, video_path=None, class_names=None, write_path=None, start_frame=0):

    video = cv2.VideoCapture(video_path)
    count = 0
    success = True
    #box_color_id = [0]
    rep_store = []
    # For video, we wish classes keep the same mask in frames, generate colors for masks
    #colors = visualize.random_colors(len(class_names))
    desired_classes = [class_names.index('person'), class_names.index('cup')]

    while success:
        if count >= start_frame:
            print("frame: ", count)
            success, image = video.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                # splash = color_splash(image, r['masks'])

                r = get_desired_results(r, desired_classes)

                '''splash = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                                     class_names, r['scores'], colors=colors, making_video=True)'''

                splash, rep_store = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                                                class_names, count, r['scores'], colors=None,
                                                                making_video=True, instance_segmentation=True,
                                                                score_thresh=0.9, rep_store=rep_store)
                #box_color_id = boxColorId

                cv2.imwrite(os.path.join(write_path, '%05d.jpg' % count), splash)
        count += 1


def segment_frames_GPU(model, video_path=None, class_names=None, write_path=None, start_frame=0):

    video = cv2.VideoCapture(video_path)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = 0
    success = True
    box_color_id = 0
    # For video, we wish classes keep the same mask in frames, generate colors for masks
    # colors = visualize.random_colors(len(class_names))

    while success:
        if count >= start_frame:
            #print("frame: ", count)
            # Read next image
            plt.clf()
            plt.close()
            images = np.zeros((height, width, 3, 4))

            for x in range(4):
                success, image = video.read()
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                images[:, :, :, x] = image

            if success:
                # Detect objects
                r = model.detect([images], verbose=0)
                # Color splash
                # splash = color_splash(image, r['masks'])

                '''splash = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                                     class_names, r['scores'], colors=colors, making_video=True)'''

                for k in range(4):
                    r = r[k]
                    splash, boxColorId = visualize.display_instances(images[:, :, :, k], r['rois'], r['masks'], r['class_ids'],
                                                                     class_names, count, box_color_id, r['scores'],
                                                                     colors=None, making_video=True, instance_segmentation=True,
                                                                     score_thresh=0.8)
                    box_color_id = boxColorId
                    count += 1

                    cv2.imwrite(os.path.join(write_path, 'frame%d.jpg' % count), splash)


# function to generate initial object detections used for initialisation of SiamMask
def generate_detections(frame, desired_classes=None, save_image=False, save_path=None):

    height, width, colors = frame.shape

    results = model.detect([frame], verbose=0)[0]

    if desired_classes is None:

        r = results
        count = 0
        box_color_id = 0
        splash, boxColorId = visualize.display_instances(frame, r['rois'], r['masks'], r['class_ids'],
                                                         class_names, count, box_color_id, r['scores'],
                                                         colors=None, making_video=True, instance_segmentation=True,
                                                         score_thresh=0.8)
        if save_image:

            cv2.imwrite(os.path.join(save_path, '00000detection.jpg'), splash)

        return results, boxColorId

    else:

        new_results = get_desired_results(results, desired_classes)

        r = new_results
        count = 0
        box_color_id = 0
        splash, boxColorId = visualize.display_instances(frame, r['rois'], r['masks'], r['class_ids'],
                                                         class_names, count, box_color_id, r['scores'],
                                                         colors=None, making_video=True, instance_segmentation=True,
                                                         score_thresh=0.8)

        if save_image:

            cv2.imwrite(os.path.join(save_path, '00000detection.jpg'), splash)

        return new_results, boxColorId


# function to remove detections of classes not required
def get_desired_results(results, desired_classes):

    N = 0

    for id in results['class_ids']:
        if id not in desired_classes:
            N += 1

    new_results = results
    N = np.zeros(N)
    count = 0

    for x, id in enumerate(results['class_ids']):
        if id not in desired_classes:
            N[count] = x
            count += 1

    new_results['rois'] = np.delete(new_results['rois'], N, 0)
    new_results['class_ids'] = np.delete(new_results['class_ids'], N)
    new_results['scores'] = np.delete(new_results['scores'], N)
    new_results['masks'] = np.delete(new_results['masks'], N, 2)

    return new_results


if __name__ == "__main__":

    '''save_directory = 'aistetic_segmentations/'

    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    list_frames = os.listdir('Aistetic/')

    for i, frame in enumerate(list_frames):

        try:

            frame_path = 'Aistetic/' + frame
            initial_frame = cv2.imread(frame_path)

            mRCNN_results = generate_detections(initial_frame, desired_classes=[class_names.index('person')],
                                                   save_image=True, save_path=save_directory)

        except:
            continue'''

    # MAIN FUNCTION DO NOT REMOVE
    # change video directory to desired directory for inference
    video_directory = 'Videos/'
    list_videos = os.listdir(video_directory)
    save_folder = 'Tracked_frames/'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    for i in range(len(list_videos)):

        os.mkdir('Tracked_frames/' + str(list_videos[i]) + '/')

        print(str(list_videos[i]))

        segment_video(model, video_path=video_directory + str(list_videos[i]), class_names=class_names,
                           write_path=save_folder + str(list_videos[i]) + '/', start_frame=0)

