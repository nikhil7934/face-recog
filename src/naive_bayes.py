import numpy
from PIL import Image
from numpy import mean,cov,double,cumsum,dot,linalg,array,rank,var
import os,sys,math
import matplotlib.pyplot

if len(sys.argv) != 3:
	print('Usage is WRONG, Check Once and Come Again!!')
	print('python3 naive_bayes.py <path-to-train-file> <path-to-test-file>')
	exit(0)
path_train_file = sys.argv[1]
path_test_file = sys.argv[2]

# For collecting the train images from the train-file
trfile = open(path_train_file,'r')
trdata = trfile.read().split('\n')
trimages = []
trlabels = []
for tr in trdata:
	if len(tr)!=0:
		tem = tr.strip().split(' ')
		trimages.append(tem[0])
		trlabels.append(tem[1])
	else:
		break

fea_obs = []
re_size = (32,32)
for i in range(len(trimages)):
	img = Image.open(trimages[i])
	img = img.resize(re_size, Image.ANTIALIAS)
	ima = img.convert('L')
	image = numpy.array(ima)
	image = image.flatten()
	fea_obs = numpy.append(fea_obs, image)
fea_obs = fea_obs.reshape(len(trimages), re_size[0]*re_size[1])
# PCA...
mean_data = mean(fea_obs, axis=0)
adj_data = fea_obs-mean_data
covaria = cov(adj_data.T)
e_values, e_vectors = numpy.linalg.eigh(covaria)
i = numpy.argsort(e_values)
i = i[::-1]
e_vectors = e_vectors[:,i]
dimensions = 32
pri_eivec = e_vectors[:,:dimensions]
pro_data = numpy.matmul(adj_data, pri_eivec)
imgclass = {}
for i in range(len(trlabels)):
	if trlabels[i] not in imgclass:
		imgclass[trlabels[i]] = {}
		imgclass[trlabels[i]]['count'] = 1
		imgclass[trlabels[i]]['image'] = []
		imgclass[trlabels[i]]['image']=numpy.append(imgclass[trlabels[i]]['image'],pro_data[i])
	else:
		imgclass[trlabels[i]]['count'] += 1
		imgclass[trlabels[i]]['image']=numpy.append(imgclass[trlabels[i]]['image'],pro_data[i])
for i in imgclass.keys():
	tem = imgclass[i]['count']
	imgclass[i]['image']=imgclass[i]['image'].reshape(tem, 32)
	imgclass[i]['mean']=numpy.mean(imgclass[i]['image'],axis=0)
	imgclass[i]['vari']=numpy.var(imgclass[i]['image'],axis=0)
	imgclass[i]['prob']=tem/len(trimages) 


# For collecting the test images from the test-file
tefile = open(path_test_file,'r')
tedata = tefile.read().split('\n')
tedata = [f for f in tedata if len(f)!=0]
for iy in tedata:
	img = Image.open(iy)
	img = img.resize(re_size,Image.ANTIALIAS)
	ima = img.convert('L')
	image = numpy.array(ima)
	image = image.flatten()
	proj_data = numpy.matmul(image, pri_eivec)
	classified_class=''
	maxi=0
	for cl in imgclass.keys():
		proj_data= numpy.real(proj_data)
		me = numpy.real(imgclass[cl]['mean'])
		va = numpy.real(imgclass[cl]['vari'])
		prob = 0
		for i in range(32):
			vale = (1/math.sqrt(2*3.14*va[i]))
			try:
				val=math.exp(-((proj_data[i]-me[i])*(proj_data[i]-me[i]))/(2*va[i]))
			except OverflowError:
				val=double('inf')
			prob = prob +val*vale
		if(maxi < prob):
			maxi=prob
			classified_class=cl
	print(classified_class)
