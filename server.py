import multiprocessing
import socket
import sys
import json
import numpy as np
import threading

from numpy.core._multiarray_umath import ndarray

import detect_face
import io
import base64
import skimage
import matplotlib.pyplot as plt
import PIL.Image
import joblib
import warnings
import face_recognition
import dlib
from sklearn.cluster import DBSCAN
import json
import matplotlib.image as mpimg
from skimage import color,feature,transform
import firebase_admin
from firebase_admin import credentials, firestore

shaper = dlib.shape_predictor(r"models\shape_predictor_5_face_landmarks.dat")
encoder = dlib.face_recognition_model_v1(r'models\dlib_face_recognition_resnet_model_v1.dat')


ERROR = "NACK".encode()
OK = "ACK".encode()


def encoding(image,boxes, face_size):
    # TODO: check rectangle x and y
    print(2)
    x, y, h, w = boxes[0], boxes[1], face_size[0], face_size[1]
    print(x, y, h, w)
    # long left, long top, long right, long bottom
    # TODO: check if right positions
    face_locations = dlib.rectangle(x,y, x+w,y+h)
    print(face_locations)
    pose_predictor_5_point = shaper
    image = image.astype(np.uint8)

    landmarks = pose_predictor_5_point(image, face_locations)
    print(3)
    face_encoder = encoder
    encodings = np.array(face_encoder.compute_face_descriptor(image, landmarks, 1))
    return encodings

def detect_and_encode(slow, fast, base64_img):
    print(base64_img)
    img = base64.decodebytes(base64_img)
    img = io.BytesIO(img)
    img = mpimg.imread(img, format='JPEG')
    img = skimage.color.rgb2grey(img)
    img = skimage.transform.rescale(img, 0.25)

    print(img.shape)
    face_size = (62, 47)

    boxes = detect_face.test(slow, fast, face_size, img)
    boxes = boxes[0]
    if len(boxes) == 0:
        return None,"no bounding box in image!"
    encoded_face = encoding(skimage.color.grey2rgb(img), boxes, face_size)
    print(encoded_face)
    return encoded_face, boxes

def scan_face(base64_img, slow, fast, users):
    buffer = base64_img  # base64 of new pic
    img_encodings, boxes = detect_and_encode(slow, fast, buffer)

    encodings = []

    for user in users:
        encodings.append(user["enc1"])
        encodings.append(user["enc2"])
        encodings.append(user["enc3"])
        encodings.append(user["enc4"])
        encodings.append(user["enc5"])
    encodings.append(img_encodings)
    print("[INFO] clustering...")
    clt = DBSCAN()
    labels = clt.fit_predict(encodings)

    labelIDs = np.unique(labels)  # [1,1,1,2,2,3,1] -> [1,2,3]
    numUniqueFaces = len(np.where(labelIDs > -1)[0])
    print("[INFO] # unique faces: {}".format(numUniqueFaces))

    for labelID in labelIDs:
        idxs = np.where(labels == labelID)[0]
        if idxs[-1] == len(encodings):  # len(encodings) = last object
            return users[idxs[0]]["name"], boxes
    return "unknown", boxes

def add_face(sock, slow, fast):
    for i in range(5):
        buffer = sock.recv(1000000)
        while buffer.decode()[-3:] != "bay":
            buffer += sock.recv(1000000)
        print(len(buffer))
        encoded_face, boxes = detect_and_encode(slow, fast, buffer[:-3])
        if type(encoded_face) is None:
            sock.sendall(ERROR + str(boxes).encode())
        sock.sendall(str(encoded_face).encode())
    return encoded_face

def server(sock, slow, fast):
    while True:
        try:
            choice = sock.recv(1024).decode()  # 0 or 1
            sock.sendall(OK)  # ACK
            if choice == '0':
                """buffer = sock.recv(1000000)  # [{name:"ben",id:1,enc:HGJK}, {name:"yahel"}]
                while buffer.decode()[-3:] != "bay":
                    buffer += sock.recv(1000000)
                buffer = json.loads(buffer[:-3])
                sock.sendall(OK)  # ACK"""
                while True:
                    base64_img = sock.recv(1000000)  # [{name:"ben",id:1,enc:HGJK}, {name:"yahel"}]
                    while base64_img.decode()[-3:] != "bay":
                        base64_img += sock.recv(1000000)
                    base64_img = base64_img[:-3]
                    if base64_img.decode() == "end":
                        break
                    boxes, name = scan_face(base64_img, slow, fast, buffer)
                    sock.sendall(str([[str(boxes[0]),str(boxes[1]),str(47),str(62),str(name)]]).encode())  # "[["x","y","w","h","name"],[],[]]"
            elif choice == '1':
                add_face(sock, slow, fast)

            else:
                raise Exception("no choice found!")
        except Exception as e:
            print(e)
            sock.sendall(ERROR+','.encode()+e.__repr__().encode())


def connect_to_firebase():
    cred = credentials.Certificate(r"face-rec-8d1ad-firebase-adminsdk-9fian-7442d9be64.json")
    app = firebase_admin.initialize_app(cred)
    db = firestore.client()
    users_ref = db.collection("users")
    docs = users_ref.stream()
    return docs


def main():
    # Create a TCP/IP socket
    docs = connect_to_firebase()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('127.0.0.1', 6135)
    sock.bind(server_address)
    # Listen for incoming connections
    sock.listen(1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fast = joblib.load(r"models\model_fast.pickle")
        slow = joblib.load(r"models\model5.pickle")

        # Wait for a connection

    print("start lisennig")
    connection, client_address = sock.accept()
    server(connection, slow, fast, docs)


if __name__ == "__main__":
    main()