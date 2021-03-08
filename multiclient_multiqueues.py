#handles multiple request at a single time
#each ML client will run requests in seperate queues so multiple clients can operate

import requests
import json
import time
import threading
from ml import Inference
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import sys

import random


print("ml backend loading...")
ml = Inference()
print("ml backend loaded...")



mlclientId = random.randint(1,10000)#hashlib.sha256(str(uuid4().hex).encode()).hexdigest()[:6]

#if len(sys.argv) < 2:
#	print("enter a client id:")
#	exit
#mlclientId = sys.argv[1]


def mlprocess(mlclientId):

	req = {"id":mlclientId}
	#res  = requests.post('http://192.168.0.204:8000/__getrequestimages',json=req)
	res  = requests.post('https://esti-mate.ml/__getrequestimages',json=req)

	if res.ok:

		print("ok");
		rlist = res.json()	#r is a list

		reqlist = [] #reply

		for r in rlist:
			nonce = r['nonce']
			url = r['url']
			#print("nonce of request received:{} with url {}".format(nonce,url))
			#print("request {} received".format(nonce))
			logging.info("request {} received".format(nonce))

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
				
			#print("filter={},damage={},frcnn={}".format(filter,damage,frcnn))
			logging.info("filter={},damage={},frcnn={}".format(filter,damage,frcnn))

			#upload the ml data
			req = {"nonce":nonce,"analysis":{"filter":filter,"damage":damage,"frcnn":frcnn}}
			reqlist.append(req)

		req  = {"id":mlclientId,"reqlist":reqlist}
		#res  = requests.post('http://192.168.0.204:8000/__uploadimagemlanalysis',json=req)
		res  = requests.post('https://esti-mate.ml/__uploadimagemlanalysis',json=req)
	else:
		None
		#print("waiting..")
		
	return

#register the mlclient
#res  = requests.post('http://192.168.0.204:8000/__registermlclient',json={"id":mlclientId})
res  = requests.post('https://esti-mate.ml/__registermlclient',json={"id":mlclientId})
if not res.ok:
	logging.error("ML client registration id {}".format(mlclientId))
	exit

time.sleep(2)

#for i in range(1000):
while True:
	mlprocess(mlclientId)
	time.sleep(5)
