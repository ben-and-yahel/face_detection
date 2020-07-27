import time
import matplotlib.pyplot as plt
import os
import numpy as np
from time import time
from skimage import color, feature
from skimage import transform
import skimage
import joblib
import threading
import multiprocessing
import matplotlib.image as mpimg

def merge_boxes(indices):
    avg_x = 0.0
    avg_y = 0.0
    for x in indices:
        avg_x += x[0]
        avg_y += x[1]
    try:
        avg_x = avg_x / len(indices)
        avg_y = avg_y / len(indices)
    except ZeroDivisionError:
        return -1, -1
    return int(avg_x), int(avg_y)


def my_nms(indices, height, width, max_distance):
    # TODO : make max distance a percent num rather that pixel distance
    indies = indices[0][0], indices[0][1]
    splits = [[]]
    counter = 0
    for patch in indices:
        if abs(indies[1] - patch[1]) > max_distance or abs(indies[0] - patch[0]) > max_distance:
           splits.append([patch])
           counter += 1
        else:
            splits[counter].append(indies)
        indies = patch[0], patch[1]
    final_avg = []
    for split in splits:
        final_avg.append((merge_boxes(split)))
    print(len(splits))
    return final_avg


def worker(patches):
    return [feature.hog(patch) for patch in patches]


def sliding_window(img, patch_size=(62,47), istep=3, jstep=3, scale=1.0):
    Ni, Nj = (int(scale * s) for s in patch_size)
    for i in range(0, img.shape[0] - Ni, istep):
        for j in range(0, img.shape[1] - Nj, jstep):
            patch = img[i:i + Ni, j:j + Nj]
            if scale != 1:
                patch = transform.resize(patch, patch_size)
            yield (i, j), patch


def dynamic_window(test_image, fast_model, slow):
    img, img_size, img_jump = test_image,0.25, 0.05
    labels = np.zeros(0)
    steps, step_jump, flag = 7, 1, True
    while labels.sum() == 0.0:
        #TODO: change to cnn
        img = skimage.transform.rescale(test_image, img_size)

        t = time()
        indices, patches = zip(*sliding_window(img, istep=steps, jstep=steps))
        print("[INFO] sliding window took: {:.2f} seconds".format(time() - t))

        t = time()
        patches_hog = np.array([feature.hog(patch) for patch in patches])
        print("[INFO] hog patches took: {:.2f} seconds".format(time() - t))

        t = time()
        fast_labels = np.array(fast_model.predict(patches_hog))
        if fast_labels.sum() == 0.0:
            steps -= 1
            continue
        print("[INFO] fast predict took: {:.2f} seconds".format(time() - t))
        print(fast_labels.sum())

        patches_hog = patches_hog[fast_labels == 1]
        indices = np.array(indices)
        indices = indices[fast_labels == 1]

        t = time()
        # patchs = patchs.reshape(-1, 62, 47, 1)
        labels = slow.predict(patches_hog)

        # labels = np.array([p[0] for p in prediction])
        print("[INFO] cnn predict took: {:.2f} seconds".format(time() - t))
        img_size -= img_jump
        if flag:
            steps -= step_jump
            flag = False
        else:
            flag = True
    indices = indices[labels == 1]
    return indices, img_size+img_jump
    #return labels, indices, patches_hog, img


def test(model, fast_model, face_patch_size, image):

    #test_image = plt.imread(image)
    #test_image = skimage.data.astronaut()
    #test_image = skimage.color.rgb2gray(test_image)
    test_image = image
    try:
        t = time()
        indices, img_size = dynamic_window(test_image, fast_model, model)
        #fast_labels, indices, patches_hog, test_image = dynamic_window(test_image, fast_model, model)
    except ValueError as e:
        print("No bounding box!",e)
        return []
    if indices.sum() == 0.0:
        raise (Exception("No bounding box!"))
    print("[INFO] dynamic window took: {:.2f} seconds".format(time()-t))

    boxes = my_nms(indices, 62, 47, 15)  # returns array with x,y position of faces
    return boxes,img_size
    #show result
    """Ni, Nj = face_patch_size
    fig, ax = plt.subplots()
    ax.imshow(test_image, cmap='gray')

    for i, j in boxes:
            ax.add_patch(plt.Rectangle((j, i), Nj, Ni, edgecolor='blue',
                                       alpha=0.3, lw=2, facecolor='none'))
    plt.waitforbuttonpress()"""


def main():
    path = r"C:\Users\user\Desktop\final project\FaceRecognition\face-clustering\test"
    img = r"\WhatsApp Image 2020-04-08 at 13.35.55 (5)"
    fast_model = joblib.load(r"models\model_fast.pickle")
    slow = joblib.load(r"models\model5.pickle")
    test(slow,fast_model, (62, 47), path+img+".jpeg")


if __name__ == '__main__':
    main()