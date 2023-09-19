import yaml
import cv2
import numpy as np
import pybboxes as pbx
import matplotlib.pyplot as plt
import torch
from torch import optim, nn
from torchvision import models, transforms
from os import listdir, mkdir
from os.path import isfile, join, splitext, basename, exists
from sklearn.cluster import KMeans



class FeatureExtractor(nn.Module):
  def __init__(self, model):
    super(FeatureExtractor, self).__init__()
		# Extract VGG-16 Feature Layers
    self.features = list(model.features)
    self.features = nn.Sequential(*self.features)
		# Extract VGG-16 Average Pooling Layer
    self.pooling = model.avgpool
		# Convert the image into one-dimensional vector
    self.flatten = nn.Flatten()
		# Extract the first part of fully-connected layer from VGG16
    self.fc = model.classifier[0]
  
  def forward(self, x):
		# It will take the input 'x' until it returns the feature vector called 'out'
    out = self.features(x)
    out = self.pooling(out)
    out = self.flatten(out)
    out = self.fc(out) 
    return out 



# Transform the image, so it becomes readable with the model
transform = transforms.Compose([
  transforms.ToPILImage(),
  transforms.CenterCrop(512),
  transforms.Resize(448),
  transforms.ToTensor()                              
])

def extract_yolo_detections(path: str):
	dirs = ['train', 'test', 'valid']
	base = basename(path)

	config = ''
	with open(f'{path}/data.yaml') as f:
		config = yaml.safe_load(f)
	
	#mkdir(f'{base}_roi')

	result = []
	for d in dirs:
		url = '{path}/{dataset}/{source}/{file}'
		images = listdir(f'{path}/{d}/images')
		for n in images:
			image = cv2.imread(url.format(path=path, dataset=d, source='images', file=n))
			labels = parse_yolo_labels(url.format(path=path, dataset=d, source='labels', file=f'{splitext(n)[0]}.txt'))
			for i, label in enumerate(labels):
				roi = extract_roi(image, label[1:])
				result.append((f'{basename(n)}_{i}.jpg', roi))
				#cv2.imwrite(f'{base}_roi/{basename(n)}_{i}.jpg', roi)
	return result
		
def parse_yolo_labels(path: str):
	with open(path, 'r') as reader:
		return [tuple(map(float, n.rstrip().split(' '))) for n in reader]

def extract_roi(image, label):
	H, W, _ = image.shape
	roi = pbx.convert_bbox(label, from_type='yolo', to_type='voc', image_size=(W, H))
	return image[roi[1]:roi[3], roi[0]:roi[2]]

def create_model():
	# Initialize the model
	model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
	new_model = FeatureExtractor(model)

	# Change the device to GPU
	#device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
	#new_model = new_model.to(device)
	return model

if __name__ == '__main__':
	model = create_model()
	data = extract_yolo_detections('./tft-objects-26')
	images = [n[1] for n in data]

	features = []
	for image in images:
		#plt.imshow(image)
		#plt.show()
		image = transform(image)
		image = image.reshape(1, 3, 448, 448)
		with torch.no_grad():
			feature = model(image)
		features.append(feature.numpy().reshape(-1))
	
	features = np.array(features)

	model = KMeans(n_clusters=100, random_state=42)
	model.fit(features)
	labels = model.labels_
	
	mkdir('cluster')
	for i, label in enumerate(labels):
		if not exists(f'cluster/{label}'):
			mkdir(f'cluster/{label}')
		cv2.imwrite(f'cluster/{label}/{data[i][0]}', data[i][1])