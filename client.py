import requests
import json
import time
import threading
from ml import Inference
import os

ml = Inference()

def mlprocess():

	req = {}
	res  = requests.post('http://192.168.0.204:8000/__getrequestimages',json=req)

	if res.ok:
		print("ok");
		r = res.json()
		nonce = r['nonce']
		url = r['url']
		print("nonce of request received:{} with url {}".format(nonce,url))

		#do processing
		#copy images
		imagefile = requests.get(url)
		filename="{}.jpg".format(nonce)
		open('images/{}'.format(filename),'wb').write(imagefile.content)
		photopath="./images/{}".format(filename)

		filter=""
		damage=""
		frcnn=""

		if(ml.Filter(photopath)):
			filter="OK"
			damage=ml.Vgg(photopath)
			frcnn=ml.Frcnn(photopath)
			
		print("filter={},damage={},frcnn={}".format(filter,damage,frcnn))


		#upload the ml data
		req = {"filter":filter,"damage":damage,"frcnn":frcnn}
		res  = requests.post('http://192.168.0.204:8000/__uploadimagemlanalysis',json=req)
		
	else:
		print("__getrequestimages failed")
		
	return

for i in range(1000):
	mlprocess()
	time.sleep(1)
