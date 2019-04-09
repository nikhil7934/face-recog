import numpy 
from PIL import Image
from numpy import mean,cov,double,cumsum,dot,linalg,array,rank
import os,sys,math
import matplotlib.pyplot

if len(sys.argv) != 3:
	perror('Usage is WRONG, Check Once and Come Again!!')
	print('python3 linear_classifier.py <path-to-train-file> <path-to-test-file>')
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
covaria = numpy.matmul(adj_data.T, adj_data)
e_values, e_vectors = numpy.linalg.eigh(covaria)
i = numpy.argsort(e_values)
i = i[::-1]
e_vectors = e_vectors[:,i]
dimensions = 32
pri_eivec = e_vectors[:,:dimensions]
pro_data = numpy.matmul(adj_data, pri_eivec)
imgclass = {}
inity = 0
for i in range(len(trlabels)):
	if trlabels[i] not in imgclass.keys():
		imgclass[trlabels[i]] = {}
		imgclass[trlabels[i]]['name'] = trlabels[i]
		imgclass[trlabels[i]]['count'] = 1
		imgclass[trlabels[i]]['image'] = []
		imgclass[trlabels[i]]['value'] = inity
		inity = inity + 1
		imgclass[trlabels[i]]['image']=numpy.append(imgclass[trlabels[i]]['image'],pro_data[i])
	else:
		imgclass[trlabels[i]]['count'] += 1
		imgclass[trlabels[i]]['image']=numpy.append(imgclass[trlabels[i]]['image'],pro_data[i])

pro_data = pro_data.T
no_of_iters = 10000
learning_rate_eta = 0.1
regularization = 100
total_classes = len(imgclass.keys())
w_mat = numpy.random.randn(total_classes, pro_data.shape[0])*0.001
t = numpy.zeros((total_classes, pro_data.shape[1]))
for i in range(len(trlabels)):
	t[imgclass[trlabels[i]]['value']][i] = 1

for i in range(no_of_iters):
	sc = numpy.matmul(w_mat, pro_data)
	sc_max = numpy.max(sc,axis=0)
	sc_exp = numpy.exp(sc-sc_max)
	sc_exp_sum = numpy.sum(numpy.exp(sc-sc_max), axis=0)
	sc_exp_nor = sc_exp/sc_exp_sum
	s1,s2 = pro_data.shape
	sc_exp_nor = t-sc_exp_nor
	gd = numpy.matmul(sc_exp_nor,pro_data.T)
	w_mat = w_mat + learning_rate_eta*gd



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
	te_sc = numpy.dot(w_mat,proj_data.T)
	cl_na = numpy.argmax(te_sc,axis=0)
	for i in imgclass.keys():
		if imgclass[i]['value'] == cl_na:
			print(imgclass[i]['name'])
			break