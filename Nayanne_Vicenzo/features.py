# -*- coding: utf-8 -*-
import sys
import numpy as np
import skimage.io as imgio
import os
import csv

from skimage import color
from scipy.stats import describe
from sklearn.neural_network import MLPClassifier

def descrever(nomeDoCSV, imClass, data):
	csv_file = open(nomeDoCSV, 'wb')
	writer = csv.writer(csv_file)
	i = 0
	writer.writerow(["Mean", "Variance", "Skewness", "Kurtosis"])
	
	# Se a imagem for colorida, converter para cinza
	for image in imClass:
		image = color.rgb2gray(image)
		# Estatística
		stats = describe(image, axis=None)
		# Resultado
		writer.writerow([stats.mean, stats.variance, stats.skewness, stats.kurtosis])
		data.append([stats.mean, stats.variance, stats.skewness, stats.kurtosis])
		i = i + 1

def buscarArquivos(path, imclass):
	# Busca todos arquivos na pasta
	for file in os.listdir(path):
		imclass.append(imgio.imread(os.path.join(path, file)))

		
class0 = "/home/grv/Projetos/Osteoporose/TCB_Challenge_Data/TCB_Challenge_Data/TRAIN_TEST_Data/Class0"
class1 = "/home/grv/Projetos/Osteoporose/TCB_Challenge_Data/TCB_Challenge_Data/TRAIN_TEST_Data/Class1"
blind = "/home/grv/Projetos/Osteoporose/TCB_Challenge_Data/TCB_Challenge_Data/BLIND_Data"

imClass0 = []
imClass1 = []
imBlind = []

data = []
target = []
pData = []

buscarArquivos(class0, imClass0)
buscarArquivos(class1, imClass1)
buscarArquivos(blind, imBlind)

print("Descrevendo a Classe 0...")
descrever("classe0.csv", imClass0, data)
for i in range(0, 58):
	target.append(0)
print("Descrevendo a Classe 1...")
descrever("classe1.csv", imClass1, data)
for i in range(0, 58):
	target.append(1)
print("Descrevendo a Blind...")
descrever("blind.csv", imBlind, pData)

data = np.asarray(data)
target = np.asarray(target)
pData = np.asarray(pData)

print(data.shape)
print(target.shape)
print(pData.shape)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1)

print(clf.fit(data, target))

print(clf.predict(pData))
