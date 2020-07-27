import face_recognition
import pickle
import os
import dlib
import numpy as np
import matplotlib.pyplot as plt
import face_recognition_models
#import open_svm


# path to input directory of faces + images
dataset = "dataset"
# path to serialized db of facial encodings
encoding_save = "encodings.pickle"
# TODO: changing openSVM func once yahel finish it

print("[INFO] quantifying faces...")
imagePaths = list(os.listdir(dataset))
data = []
model_path = "model5"

def main():
	# loop over the image paths
	for (i, imagePath) in enumerate(imagePaths):
		if i < 3:
			print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
			imagePath = dataset+'/'+imagePath
			print(imagePath)
			image = plt.imread(imagePath)

			# detect the (x, y)-coordinates of the bounding boxes
			# corresponding to each face in the input image
			try:
				boxes = face_recognition.face_locations(image, model='hog')  # bounding box
			except:
				continue
			#boxes = open_svm.getXY(image)

			# compute the facial embedding for the face
			face_locations = [dlib.rectangle(face_location[3], face_location[0], face_location[1], face_location[2]) for face_location in boxes]
			pose_predictor_68_point = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
			landmarks = [pose_predictor_68_point(image, face_location) for face_location in face_locations]
			face_encoder = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
			encodings = [np.array(face_encoder.compute_face_descriptor(image, raw_landmark_set, 1)) for raw_landmark_set in landmarks]

			# build a dictionary of the image path, bounding box location,
			# and facial encodings for the current image
			d = [{"imagePath": imagePath, "loc": box, "encoding": enc} for (box, enc) in zip(boxes, encodings)]
			data.extend(d)
			# print(d)

		# print(data)
	print(data)
	# dump the facial encodings data to disk
	print("[INFO] serializing encodings...")
	f = open(encoding_save, "wb")
	print(len(pickle.dumps(data)))
	f.write(pickle.dumps(data))

	f.close()


if __name__ == '__main__':
	main()
