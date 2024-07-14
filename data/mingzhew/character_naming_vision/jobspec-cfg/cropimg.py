# read annotation and crop face images from videos.
# store in h5
import os
import numpy as np
import h5py
import cv2
import sys
from imageio import imsave
import pickle
from PIL import Image

mvdir = '/scratch/jiadeng_flux/mzwang/MovieQA_benchmark/story/video_clips/'
trkdir = '/scratch/jiadeng_flux/mzwang/face_tracks/'
outdir = '/scratch/jiadeng_fluxoe/mzwang/mvqa/cropimg_all'
if os.path.exists(outdir) == False:
	os.mkdir(outdir)
#fid = open('mvlist.txt')
#mvlist = fid.readlines()
mvlist = [sys.argv[1]]
#fid.close()
for i in mvlist:
	mv = i.strip()
	vidlist = os.listdir(os.path.join(mvdir , mv))
	outpath = os.path.join(outdir , mv)
	if os.path.exists(outpath) == False:
		os.mkdir(outpath)
	for vid in vidlist:	
		fid = open(os.path.join(trkdir , mv , vid)+'.pi')		
		trks = pickle.load(fid)
		fid.close()
		num = len(trks)
		mvfaces = np.ndarray((num*15,224,224,3))
		mvidx = np.ndarray(num)
		print vid
		cap = cv2.VideoCapture(os.path.join(mvdir , mv , vid))	
		frame = []
		while cap.isOpened():
			ret , f = cap.read()
			if ret == True:
				frame.append(f)
			else:
				break
		cnt = 0
		for k,trk in enumerate(trks):
			j = 0
			while True:
				img = Image.fromarray(frame[j*5+trk[0]])
				[x1,y1,x2,y2] = [int(round(trk[1][j*5,0])),int(round(trk[1][j*5,1])),int(round(trk[1][j*5,2])),int(round(trk[1][j*5,3]))]
				x1 = int(max(x1 , 0))
				y1 = int(max(y1,  0))
				x2 = int(min(x2 , frame[0].shape[1]-1))
				y2 = int(min(y2,  frame[0].shape[0]-1))
				face = img.crop((x1,y1,x2,y2))
				face = face.resize((224,224))
				mvfaces[cnt,:,:,:] = np.array(face.getdata()).reshape(face.size[0],face.size[1],3)
				print [k,j]
				cnt = cnt+1
				j = j+1
				if j*5 >= trk[1].shape[0]:
					break
			mvidx[k] = j
		cap.release()
		mvfaces = mvfaces[:cnt,:,:,:]
		with h5py.File(os.path.join(outpath , vid)+'.h5' , 'w') as hf:
			hf.create_dataset('faces' , data=mvfaces)
			hf.create_dataset('idx' , data=mvidx)
