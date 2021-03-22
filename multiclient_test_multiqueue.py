#handles multiple request at a single time

import requests
import json
import time
import threading
from ml import Inference
import os
#import logging
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import sys
import random
import socket 

print("ml backend loading...")
ml = Inference()
print("ml backend loaded...")

import socket
h_name = socket.gethostname()
IP_address = socket.gethostbyname(h_name)

mlclientId = random.randint(1,10000)#hashlib.sha256(str(uuid4().hex).encode()).hexdigest()[:6]
basedir = os.path.abspath(os.path.dirname(__file__))


def writestat(mlclientId,ip,clienttype,requests_no,estimates_no,rejects_no):
	'''write the stats to the stat-monitor system at port 8001 in the local host'''
	req = {"id":mlclientId,"stats":{"ip":ip,"type":clienttype,"requests":requests_no,"estimates":estimates_no,"rejects":rejects_no}}
	try:
		res  = requests.post('http://192.168.0.204:8001/__sendstats',json=req)
	except:
		None
	return
	
def savestats(log_file_path,request_no,estimates_no,rejects_no):
	f = open(log_file_path,"w")
	str = "{},{},{}".format(requests_no,estimates_no,rejects_no)
	f.write(str)
	f.close()
	return


def mlprocess(mlclientId,requests_no,estimates_no,rejects_no):

	req = {"id":mlclientId}
	res  = requests.post('http://192.168.0.204:8000/__getrequestimages',json=req)
	#res  = requests.post('https://esti-mate.ml/__getrequestimages',json=req)

	if res.ok:

		print("ok");
		rlist = res.json()	#r is a list

		reqlist = [] #reply

		for r in rlist:
			nonce = r['nonce']
			url = r['url']

			#increment request
			requests_no +=1

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
				estimates_no += 1
			else:
				'''reject'''
				rejects_no +=1
				

			#upload the ml data
			req = {"nonce":nonce,"analysis":{"filter":filter,"damage":damage,"frcnn":frcnn}}
			reqlist.append(req)

		req  = {"id":mlclientId,"reqlist":reqlist}
		res  = requests.post('http://192.168.0.204:8000/__uploadimagemlanalysis',json=req)
		#res  = requests.post('https://esti-mate.ml/__uploadimagemlanalysis',json=req)
	else:
		None
		#print("waiting..")
	return requests_no, estimates_no, rejects_no





if __name__ == "__main__":
	#register the mlclient
	res  = requests.post('http://192.168.0.204:8000/__registermlclient',json={"id":mlclientId})
	if not res.ok:
		print("ML client {} registration error in the test esti server at port 8000".format(mlclientId))
		exit

	#register in to the stat server 
	try:
		res  = requests.post('http://192.168.0.204:8001/__registermlclient',json={"id":mlclientId}) #type = test or production server
		if not res.ok:
			print("ML client id {} registration error in the stat server at port 8001".format(mlclientId))
			#exit
			pass
	except:
		pass

	time.sleep(5)
	
	#read a simple log file to get the total requests,estimates,rejects
	requests_no,estimates_no,rejects_no = 0,0,0

	log_file_path = os.path.join(basedir,"stats/log.txt")
	try:
		f = open(log_file_path,"r+")
		line=f.readline()
		requests_no,estimates_no,rejects_no = line.split(',')
		print("starting req: {} esti: {} reject: {}".format(requests_no,estimates_no,rejects_no))
		f.close()
	except:
		savestats(log_file_path,0,0,0)
		


	while True:
		try:
			requests_no, estimates_no, rejects_no = mlprocess(mlclientId,int(requests_no),int(estimates_no),int(rejects_no))
			time.sleep(2)
			writestat(mlclientId,IP_address,"TEST",requests_no,estimates_no,rejects_no)
			savestats(log_file_path,requests_no,estimates_no,rejects_no)
		except KeyboardInterrupt:
			print("CTRL-C pressed exiting..")
			savestats(log_file_path,requests_no,estimates_no,rejects_no)
			sys.exit()
