import numpy as np
from time import time
from skimage import data, color, feature
from sklearn.feature_extraction.image import PatchExtractor
from skimage import transform
import skimage
from sklearn.datasets import fetch_lfw_people
import joblib
from itertools import chain
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
"""all algorithms"""
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV


def split_four(test_image):
    test_image = skimage.transform.rescale(test_image, 2)
    height, width = int(test_image.shape[0]/2), int(test_image.shape[1]/2)
    pics = np.zeros((4,height,width))
    pics[0] = test_image[:height,:width]
    pics[1] = test_image[height:,:width]
    pics[2] = test_image[:height,width:]
    pics[3] = test_image[height:,width:]
    """
    for im in pics:
    fig, ax = plt.subplots()
    ax.imshow(im, cmap='gray')
    ax.axis('off')"""
    return pics


def fetch_positive_objects():
    faces = fetch_lfw_people(download_if_missing=True)
    positive_patches = faces.images[:int(len(faces.images) / 10) * 9]
    extra_patches = faces.images[int(len(faces.images) / 10) * 9:]
    return positive_patches, extra_patches


def fetch_negative_objects():
    imgs_to_use = ['camera', 'text', 'coins', 'moon',
                   'page', 'clock', 'immunohistochemistry',
                   'chelsea', 'coffee', 'hubble_deep_field']
    images = [skimage.color.rgb2gray(getattr(data, name)()) for name in imgs_to_use]  # data = skimage.data
    return images


def extract_patches(img, N,  patch_size, scale=1.0):
    # takes the size we need from the negative pics
    extracted_patch_size = tuple((scale * np.array(patch_size)).astype(int))
    extractor = PatchExtractor(patch_size=extracted_patch_size,max_patches=N, random_state=0)
    patches = extractor.transform(img[np.newaxis])  # TO DO:
    if scale != 1:
        patches = np.array([transform.resize(patch, patch_size) for patch in patches])
    return patches


def combine_pos_and_neg(positive_patches, negative_patches, extra_faces):
    t = time()
    if extra_faces is None:
        X_train = np.array([feature.hog(im) for im in chain(positive_patches, negative_patches)])
    else:
        X_train = np.array([feature.hog(im) for im in chain(positive_patches, negative_patches, extra_faces)])
    y_train = np.zeros(X_train.shape[0])
    y_train[:positive_patches.shape[0]] = 1
    print("[INFO] combine took: {:.2f} seconds,google colab took:37 sec".format(time() - t))
    return X_train, y_train


def train_cnn(X, y):
    t1 = time()
    X = X / 255.0
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    print("[INFO] time took to prepare features: {}".format(time() - t1))
    t1 = time()

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/face_detection')

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=X.shape[1:]))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3)))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Activation("relu"))

    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation("sigmoid"))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.fit(X, y, batch_size=50, epochs=3, validation_split=0.1, callbacks=[tensorboard])
    print("[INFO] time took to train the model: {}".format(time() - t1))

    return model


def train_tree(X_train, y_train):
    t = time()

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=None, random_state=42)
    print("[INFO] train test splits took: {:.2f} seconds".format(time() - t))

    t = time()
    model = RandomForestClassifier(min_samples_leaf=10)  # TODO:find the best min leaf
    model.fit(X_train, y_train)
    print("[INFO] fit took: {:.2f} mins google collab did 2 mins".format(time() - t))
    return model


def main():
    print("""part 1 - fetch positive and extra""")
    images = fetch_negative_objects()
    positive_patches, extra_patches = fetch_positive_objects()
    print('faces size: ', positive_patches[0].shape)

    print("""part 2 - fetch negative""")
    # extract the needed patch from the negative images
    t = time()
    negative_patches = np.vstack([extract_patches(im, 1000,  positive_patches[0].shape, scale) for im in images for scale in [0.5, 1.0, 2.0]])
    print(negative_patches.shape, "[INFO] extract negative took: {:.2f} seconds ,google colab took:30 sec".format(time() - t))  # excepted result == (30000, 62, 47)

    print("""part 3 - split extra patches to 4""")
    # faces is the new array!
    extra_faces = np.zeros((extra_patches.shape[0] * 4, 62, 47))
    print(extra_faces.shape)
    for face in range(len(extra_patches)):
        lis = split_four(extra_patches[face])
        for i in range(4):
            extra_faces[face + i] = lis[i]

    print("""part 4 - combine both""")
    print(positive_patches.shape)
    print(negative_patches.shape)
    X_train, y_train = combine_pos_and_neg(positive_patches, negative_patches, extra_faces)

    print("""part 5 - train the model""")
    model = train_tree(X_train, y_train)

    print("""part 6 - saving""")
    joblib.dump(model, 'fast_model.pickle')  # saves the model

    print("""part 7 - combine features for cnn""")
    X_train, y_train = combine_pos_and_neg(positive_patches, negative_patches,None)

    print("""part 8 - train the model""")
    cnn = train_cnn(X_train, y_train)

    print("""part 9 - save the cnn""")
    cnn.save('face_detection.model')


if __name__ == '__main__':
    main()
