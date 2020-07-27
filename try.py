from time import time
import skimage
from skimage import feature,color,transform
import multiprocessing
import numpy as np
import joblib
import warnings
from multiprocessing import Process, Queue
import difflib, random
import firebase_admin
from firebase_admin import credentials, firestore


if __name__ == '__main__':
    cred = credentials.Certificate(r"face-rec-8d1ad-firebase-adminsdk-9fian-7442d9be64.json")
    app = firebase_admin.initialize_app(cred)
    db = firestore.client()
    users_ref = db.collection("users")
    docs = users_ref.stream()

    #for i, doc in enumerate(docs):
    #print(u'{} => {}'.format(docs[].id, doc.to_dict()))

    doc_ref = db.collection("users").document(str(int(list(docs)[-1].id)+1))
    doc_ref.set({'name': "yahel"})