# Pierre To (1734636), Jérémie Jasmin (1800865), Guillaume Vergnolle (1968693)
# INF8225 - Projet
# Base sur : https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py
# Base de donnees : https://keras.io/datasets/#cifar10-small-image-classification

from keras.datasets import cifar10

import random
import matplotlib.pyplot as plt

import os
import sys
import numpy as np

# Fonction qui crée une image d'échantillon
def sample_images(set, epoch):
    r, c = 5, 5
    
    print(set.shape())

    # Redimensionnement de l'image à 0 - 1
    #gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    for i in range(r):
        for j in range(c):

            index = random.randint(0,set.shape()[0])
            img = set[index, :, :, :]

            if self.channels == 1:
                axs[i,j].imshow(img, cmap='gray')
            else:
                axs[i,j].imshow(img)
            axs[i,j].axis('off')

    fig.savefig(("{0}{1}.png").format(self.savePath, epoch))
    plt.close()

# Programme principal
if __name__ == '__main__':
    #gan = GAN()
    #gan.train(epochs=100000, batch_size=32, sample_interval=200) # epochs: 30000
    # Charger les données
    (X_train, y_train), (_, _) = cifar10.load_data()
    X_train = np.array(X_train[np.argwhere(y_train.squeeze() == 5)].squeeze())

    for i in range(0, 5):
        sample_images(X_train, epoch)