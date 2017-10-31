# -*- coding: utf-8 -*-
import sys
import numpy as np
import skimage.io as imgio
import os
import csv

from skimage import color
from scipy.stats import describe

def descrever(nomeDoCSV, imClass):
	csv_file = open(nomeDoCSV, 'wb')
	writer = csv.writer(csv_file)
	i = 0
	writer.writerow(["Numero da imagem", "Mean", "Variance", "Skewness", "Kurtosis"])
	
	# Se a imagem for colorida, converter para cinza
	for image in imClass:
		image = color.rgb2gray(image)
		# Estatística
		stats = describe(image, axis=None)
		# Resultado
		writer.writerow([i, stats.mean, stats.variance, stats.skewness, stats.kurtosis])
		i = i + 1
		

class0 = "/home/grv/Projetos/Osteoporose/TCB_Challenge_Data/TCB_Challenge_Data/TRAIN_TEST_Data/Class0"
class1 = "/home/grv/Projetos/Osteoporose/TCB_Challenge_Data/TCB_Challenge_Data/TRAIN_TEST_Data/Class1"
blind = "/home/grv/Projetos/Osteoporose/TCB_Challenge_Data/TCB_Challenge_Data/BLIND_Data"

imClass0 = []
imClass1 = []
imBlind = []

# Busca todos arquivos na pasta
for file in os.listdir(class0):
    imClass0.append(imgio.imread(os.path.join(class0, file)))

for file in os.listdir(class1):
    imClass1.append(imgio.imread(os.path.join(class1, file)))
	
for file in os.listdir(blind):
    imBlind.append(imgio.imread(os.path.join(blind, file)))
	
print("Descrevendo a Classe 0...")
descrever("classe0.csv", imClass0)
print("Descrevendo a Classe 1...")
descrever("classe1.csv", imClass1)
print("Descrevendo a Blind...")
descrever("blind.csv", imBlind)