# -*- coding: utf-8 -*-
import numpy as np
from sklearn.feature_extraction import image
from PIL import Image
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

def lerImagem(prefixo, classe):
	im = []
	if classe:
		im = Image.open('Class1\Image_1_' + str(prefixo) + '.tif')
	else:
		im = Image.open('Class0\Image_0_' + str(prefixo) + '.tif')
	
	return im.getdata()

def lerImagens(classe, imagensTreino, imagensTeste, targetTreino, targetTeste):
	for i in range(1, 30):
		imagensTreino.append(lerImagem(i, classe))
		imagensTeste.append(lerImagem(i + 29, classe))
		targetTeste.append(classe)
		targetTreino.append(classe)

def lerDados():
	imagensTreino = []
	imagensTeste = []
	targetTreino = []
	targetTeste = []
	
	lerImagens(0, imagensTreino, imagensTeste, targetTreino, targetTeste)
	lerImagens(1, imagensTreino, imagensTeste, targetTreino, targetTeste)
		
	imagensTreino = np.array(imagensTreino)
	imagensTeste = np.array(imagensTeste)
	targetTreino = np.array(targetTreino)
	targetTeste = np.array(targetTeste)
	
	imagensTreino = imagensTreino.reshape(58, 400 * 400)
	imagensTeste = imagensTeste.reshape(58, 400 * 400)
	
	return imagensTreino, imagensTeste, targetTeste, targetTreino

def avaliar(avaliador, imagensTreino, targetTreino, imagensTeste, targetTeste):
	print("Resultado com o solver " + avaliador)
	clf = MLPClassifier(solver=avaliador, alpha=1e-5, random_state=1)
	
	clf.fit(imagensTreino, targetTreino)
	
	predito = clf.predict(imagensTeste)
	print("Resultado: " + str(np.mean(predito == targetTeste)))
	
	scores = cross_val_score(clf, imagensTeste, targetTeste, cv = 3, verbose = 3, scoring='accuracy')
	print("Validacao cruzada: " + str(np.mean(scores)))

imagensTreino = []
imagensTeste = []
targetTreino = []
targetTeste = []

[imagensTreino, imagensTeste, targetTeste, targetTreino] = lerDados()

print("Imagens:")
print("\t Treino: " + str(imagensTreino.shape))
print("\t Teste: " + str(imagensTeste.shape))
print("Target:")
print("\t Treino: " + str(targetTreino.shape))
print("\t Teste: " + str(targetTeste.shape))

avaliar("lbfgs", imagensTreino, targetTreino, imagensTeste, targetTeste)
avaliar("sgd", imagensTreino, targetTreino, imagensTeste, targetTeste)
avaliar("adam", imagensTreino, targetTreino, imagensTeste, targetTeste)
