'''
	main inference code of identification and detection of vehicle damage
	
	identification 			- code under the ./bodywork folder	- vgg + custom nn
	damage area detection	- code under the ./detection folder - frcnn code
'''

from __future__ import division
	
#feature extration from VGG https://spandan-madan.github.io/DeepLearningProject/
#vgg imports
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import tensorflow as tf
import os
import pickle

#frcnn imports
import cv2
import sys
import time
from detection.config import Config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from detection import roi_helpers
from detection import resnet 
sys.setrecursionlimit(40000)

#gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
#gpu


#tdlite filter code
from tensorflow.lite.python.interpreter import Interpreter


class Inference(object):
	''' Inference class'''
	
	class FRCNNConfig(Config):
		'''specific config file for the frcnn model'''
		def __init__(self):
			Config.__init__(self)
			self.class_mapping = {'nodamage': 0, 'major': 1, 'minor': 2, 'damage': 3, 'scratch': 4, 'major"': 5, 'gun': 6, 'bg': 7}
			self.network = "resnet50"
			self.use_horizontal_flips = False
			self.use_vertical_flips = False
			self.rot_90 = False
			self.num_rois = 32
			self.model_path = "model_frcnn.hdf5"
			self.img_path = "./app/upload/carimage"

	'''helper classes for the Frncc inference'''	
	def format_img_size(self,img, C):
		""" formats the image size based on config """
		img_min_side = float(C.im_size)
		(height,width,_) = img.shape
			
		if width <= height:
			ratio = img_min_side/width
			new_height = int(ratio * height)
			new_width = int(img_min_side)
		else:
			ratio = img_min_side/height
			new_width = int(ratio * width)
			new_height = int(img_min_side)
		img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
		return img, ratio	

	def format_img_channels(self,img, C):
		""" formats the image channels based on config """
		img = img[:, :, (2, 1, 0)]
		img = img.astype(np.float32)
		img[:, :, 0] -= C.img_channel_mean[0]
		img[:, :, 1] -= C.img_channel_mean[1]
		img[:, :, 2] -= C.img_channel_mean[2]
		img /= C.img_scaling_factor
		img = np.transpose(img, (2, 0, 1))
		img = np.expand_dims(img, axis=0)
		return img

	def format_img(self,img, C):
		""" formats an image for model prediction based on config """
		img, ratio = self.format_img_size(img, C)
		img = self.format_img_channels(img, C)
		return img, ratio

	def get_real_coordinates(self,ratio, x1, y1, x2, y2):
		'''Method to transform the coordinates of the bounding box to its original size'''
		real_x1 = int(round(x1 // ratio))
		real_y1 = int(round(y1 // ratio))
		real_x2 = int(round(x2 // ratio))
		real_y2 = int(round(y2 // ratio))
		return (real_x1, real_y1, real_x2 ,real_y2)

	def loadFrcnn(self):
		'''frcnn model info load'''
		self.C = self.FRCNNConfig()

		# turn off any data augmentation at test time
		self.C.use_horizontal_flips = False
		self.C.use_vertical_flips = False
		self.C.rot_90 = False

		self.class_mapping = self.C.class_mapping

		if 'bg' not in self.class_mapping:
			self.class_mapping['bg'] = len(self.class_mapping)

		self.class_mapping = {v: k for k, v in self.class_mapping.items()}
		print(self.class_mapping)
		self.class_to_color = {self.class_mapping[v]: np.random.randint(0, 255, 3) for v in self.class_mapping}
		self.C.num_rois = 32

		if self.C.network == 'resnet50':
			self.num_features = 1024
		elif self.C.network == 'vgg':
			self.num_features = 512

		if K.image_dim_ordering() == 'th':
			self.input_shape_img = (3, None, None)
			self.input_shape_features = (num_features, None, None)
		else:
			self.input_shape_img = (None, None, 3)
			self.input_shape_features = (None, None, self.num_features)


		self.img_input = Input(shape=self.input_shape_img)
		self.roi_input = Input(shape=(self.C.num_rois, 4))
		self.feature_map_input = Input(shape=self.input_shape_features)

		# define the base network (resnet here, can be VGG, Inception, etc)
		self.shared_layers = resnet.nn_base(self.img_input, trainable=True)

		# define the RPN, built on the base layers
		self.num_anchors = len(self.C.anchor_box_scales) * len(self.C.anchor_box_ratios)
		self.rpn_layers = resnet.rpn(self.shared_layers, self.num_anchors)

		self.classifier = resnet.classifier(self.feature_map_input, self.roi_input, self.C.num_rois, nb_classes=len(self.class_mapping), trainable=True)

		self.model_rpn = Model(self.img_input, self.rpn_layers)
		self.model_classifier_only = Model([self.feature_map_input, self.roi_input], self.classifier)

		self.model_classifier = Model([self.feature_map_input, self.roi_input], self.classifier)

		print('Loading weights from {}'.format(self.C.model_path))
		self.model_rpn.load_weights(self.C.model_path, by_name=True)
		self.model_classifier.load_weights(self.C.model_path, by_name=True)
		self.model_rpn.compile(optimizer='sgd', loss='mse')
		self.model_classifier.compile(optimizer='sgd', loss='mse')

		return

	def tfliteload(self):
		'''tflite model load for filtering'''
		self.tflite_model="filter/detect.tflite"
		self.tflite_label_file="filter/labelmap.txt"
		self.tfthreshold=0.5
		with open(self.tflite_label_file,"r") as f:
			self.tflabels = [line.strip() for line in f.readlines()]
		if self.tflabels[0] == '???':
			del(self.tflabels[0])
		self.interpreter = Interpreter(model_path=self.tflite_model)
		self.interpreter.allocate_tensors()
		self.tfinput_details = self.interpreter.get_input_details()
		self.tfoutput_details = self.interpreter.get_output_details()
		self.tfheight = self.tfinput_details[0]['shape'][1]
		self.tfwidth = self.tfinput_details[0]['shape'][2]
		self.tffloating_model = (self.tfinput_details[0]['dtype'] == np.float32)
		self.tfinput_mean = 127.5
		self.tfinput_std = 127.5

		

	################################################
	def __init__(self):
		''' load all the models'''

		self.VGG = VGG16(weights='imagenet', include_top=False) #this model captures the featureas
		self.IdenticationModel = tf.keras.models.load_model('quarterpanel_damage_model.h5')
		self.VGG.summary()
		self.IdenticationModel.summary()
		self.loadFrcnn()
		self.tfliteload()
		

	def Vgg(self,_img):
		'''run the inference to recognize the image'''
		
		self.img = image.load_img(_img,target_size=(224,224))
		self.x = image.img_to_array(self.img)
		self.x = np.expand_dims(self.x, axis=0)
		self.x = preprocess_input(self.x)
		self.features = self.VGG.predict(self.x)
		self.X = self.features.reshape(1,-1)

		try:
			self.Y_preds = self.IdenticationModel.predict(self.X).squeeze()
		
			print("inference of img:{} is {}".format(self.img,self.Y_preds))
		
			#collect garbage
			del self.img, self.x, self.X, self.features

			if self.Y_preds[0]:
				print("no damage")
				del self.Y_preds
				return "nil"
			elif self.Y_preds[1]:
				print("minor")
				del self.Y_preds
				return "minor"
			elif self.Y_preds[2]:
				print("major")
				del self.Y_preds
				return "major"
			else:
				print("unclassified")
				return "unclassified"

		except Exception as e:
				return "classification errori: %s"%e


	def Frcnn(self,_img):
		'''run the inference to detect the specific damage area'''
		
		#def classify(self,img_path,img_name):
		all_imgs = []
		classes = {}
		bbox_threshold = 0.8
		visualise = True
		filepath = _img #os.path.join(self.C.img_path,_img)
		img = cv2.imread(filepath)
		X, ratio = self.format_img(img, self.C)

		if K.image_dim_ordering() == 'tf':
			X = np.transpose(X, (0, 2, 3, 1))

		# get the feature maps and output from the RPN
		[Y1, Y2, F] = self.model_rpn.predict(X)
		R = roi_helpers.rpn_to_roi(Y1, Y2, self.C, K.image_dim_ordering(), overlap_thresh=0.7)

		# convert from (x1,y1,x2,y2) to (x,y,w,h)
		R[:, 2] -= R[:, 0]
		R[:, 3] -= R[:, 1]

		# apply the spatial pyramid pooling to the proposed regions
		bboxes = {}
		probs = {}

		for jk in range(R.shape[0]//self.C.num_rois + 1):
			ROIs = np.expand_dims(R[self.C.num_rois*jk:self.C.num_rois*(jk+1), :], axis=0)
			if ROIs.shape[1] == 0:
				break

			if jk == R.shape[0]//self.C.num_rois:
				#pad R
				curr_shape = ROIs.shape
				target_shape = (curr_shape[0],self.C.num_rois,curr_shape[2])
				ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
				ROIs_padded[:, :curr_shape[1], :] = ROIs
				ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
				ROIs = ROIs_padded

			[P_cls, P_regr] = self.model_classifier_only.predict([F, ROIs])

			for ii in range(P_cls.shape[1]):

				if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
					continue

				cls_name = self.class_mapping[np.argmax(P_cls[0, ii, :])]

				if cls_name not in bboxes:
					bboxes[cls_name] = []
					probs[cls_name] = []

				(x, y, w, h) = ROIs[0, ii, :]

				cls_num = np.argmax(P_cls[0, ii, :])
				try:
					(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
					tx /= self.C.classifier_regr_std[0]
					ty /= self.C.classifier_regr_std[1]
					tw /= self.C.classifier_regr_std[2]
					th /= self.C.classifier_regr_std[3]
					x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
				except:
					pass
				bboxes[cls_name].append([self.C.rpn_stride*x, self.C.rpn_stride*y, self.C.rpn_stride*(x+w), self.C.rpn_stride*(y+h)])
				probs[cls_name].append(np.max(P_cls[0, ii, :]))

		all_dets = []

		for key in bboxes:
			bbox = np.array(bboxes[key])

			new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
			for jk in range(new_boxes.shape[0]):
				#detect only the damaged regions and how it on the images
				if key == 'nodamage':
					continue
		
				(x1, y1, x2, y2) = new_boxes[jk,:]
				(real_x1, real_y1, real_x2, real_y2) = self.get_real_coordinates(ratio, x1, y1, x2, y2)
				cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(self.class_to_color[key][0]), int(self.class_to_color[key][1]), int(self.class_to_color[key][2])),2)
				textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
				all_dets.append((key,100*new_probs[jk]))

				(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
				textOrg = (real_x1, real_y1-0)

				cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
				cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
				cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

		cv2.imwrite(filepath,img)

		if 'damage' in bboxes.keys():
			return('damage')
		elif 'major' in bboxes.keys():
			return('major')
		else:
			return('nodamage')

		return 'nodamage'


	def Filter(self,_img):
		'''filter the upload image to remove anything other than vehicles'''
		# Load image and resize to expected shape [1xHxWx3]
		image = cv2.imread(_img)
		image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		imH, imW, _ = image.shape
		image_resized = cv2.resize(image_rgb, (self.tfwidth, self.tfheight))
		input_data = np.expand_dims(image_resized, axis=0)
		# Normalize pixel values if using a floating model (i.e. if model is non-quantized)

		if self.tffloating_model:
			input_data = (np.float32(input_data) - input_mean) / input_std
		# Perform the actual detection by running the model with the image as input
		self.interpreter.set_tensor(self.tfinput_details[0]['index'], input_data)
		self.interpreter.invoke()
		
		# Retrieve detection results
		# Bounding box coordinates of detected objects
		boxes = self.interpreter.get_tensor(self.tfoutput_details[0]['index'])[0]
		classes = self.interpreter.get_tensor(self.tfoutput_details[1]['index'])[0]  # Class index of detected objects
		scores = self.interpreter.get_tensor(self.tfoutput_details[2]['index'])[0]  # Confidence of detected objects

		vehicle_select = False
		filter_class = ["truck","car","motocycle","bus"]

		# Loop over all detections and draw detection box if confidence is above minimum threshold
		for i in range(len(scores)):
			if ((scores[i] > self.tfthreshold) and (scores[i] <= 1.0)):
				if (self.tflabels[int(classes[i])] in filter_class  and (not vehicle_select)):
					print("found a vehicle class in image") 
					ymin = int(max(1,(boxes[i][0] * imH)))
					xmin = int(max(1,(boxes[i][1] * imW)))
					ymax = int(min(imH,(boxes[i][2] * imH)))
					xmax = int(min(imW,(boxes[i][3] * imW)))
					
					#	#check if person scale is comparative to the object we like to blend in. if not scale the object.
					if (ymax - ymin) > 0.1*imH:
						vehicle_select = True			
				
					# Get bounding box coordinates and draw box
					# Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
					ymin = int(max(1,(boxes[i][0] * imH)))
					xmin = int(max(1,(boxes[i][1] * imW)))
					ymax = int(min(imH,(boxes[i][2] * imH)))
					xmax = int(min(imW,(boxes[i][3] * imW)))
					
					cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

					# Draw label
					object_name = self.tflabels[int(classes[i])] # Look up object name from "labels" array using class index
					label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
					labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
					label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
					cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
					cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

					# All the results have been drawn on the image, now display the image
					#cv2.imshow('Object detector', image)
					cv2.imwrite(_img,image)
					return True

		if not vehicle_select:
			print("DID NOT find a vehicle class in image") 
			return False
