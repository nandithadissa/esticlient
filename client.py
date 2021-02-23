import requests
import json
import time
import threading
from ml import Inference
import os


print("ml backend loading...")
ml = Inference()
print("ml backend loaded...")

def mlprocess():

	req = {}
	#res  = requests.post('http://192.168.0.204:8000/__getrequestimages',json=req)
	res  = requests.post('https://esti-mate.ml/__getrequestimages',json=req)

	if res.ok:
		print("ok");
		r = res.json()
		nonce = r['nonce']
		url = r['url']
		#print("nonce of request received:{} with url {}".format(nonce,url))
		print("request {} received".format(nonce))

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
		#res  = requests.post('http://192.168.0.204:8000/__uploadimagemlanalysis',json=req)
		res  = requests.post('https://esti-mate.ml/__uploadimagemlanalysis',json=req)
	else:
		None
		#print("waiting..")
		
	return

#for i in range(1000):
while True:
	mlprocess()
	time.sleep(5)
