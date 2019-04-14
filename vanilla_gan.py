# Pierre To (1734636), Jérémie Jasmin (1800865), Guillaume Vergnolle
# INF8225 - Projet
# Base sur : https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py
# Base de donnees : https://keras.io/datasets/#cifar10-small-image-classification

from __future__ import print_function, division

from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import os
import sys
import numpy as np
import time

# Implémentation d'un GAN avec un generateur et un discriminateur selon un perceptron multi-couche
class GAN():
    def __init__(self):
        self.img_rows = 32 # Hauteur en pixels des images
        self.img_cols = 32 # Largeur en pixels des images
        self.channels = 3 # Nombre de canaux de couleur
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100 # Taille du vecteur latent z (taille de l'entrée du générateur)

        self.savePath = 'result/vanilla_gan/'
        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

        optimizer = Adam(0.0002, 0.5) # taux d'aprentissage de 0.0002 et hyperparametre Beta1 pour l'optimisateur Adam

        # Construire le discriminateur
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Construire le générateur
        self.generator = self.build_generator()

        # Le générateur prend le bruit comme entrée et génère des images
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # Pour ce modèle, on n'entraîne pas le discriminateur (seulement le générateur)
        self.discriminator.trainable = False

        # Le discriminateur prend les images générées comme entrée et détermine la validité
        validity = self.discriminator(img)

        # Modèle combiné (pile de générateur et de générateur)
        # Entraînement du générateur pour tromper le discriminateur
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    # Fonction qui construit le générateur
    # Fonction d'activation : tanh
    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        # Sauvegarde du générateur dans un fichier
        with open(self.savePath + 'generator.txt','w+') as fh:
            model.summary(print_fn=lambda x: fh.write(x + '\n'))

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    # Fonction qui construit le discriminateur
    # Fonction d'activation : sigmoid
    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        # Sauvegarde du discriminateur dans un fichier
        with open(self.savePath + 'discriminator.txt','w+') as fh:
            model.summary(print_fn=lambda x: fh.write(x + '\n'))

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    # Entraînement
    def train(self, epochs, batch_size=128, sample_interval=50):

        # Charger les données
        (X_train, y_train), (_, _) = cifar10.load_data()
        X_train = np.array(X_train[np.argwhere(y_train.squeeze() == 5)].squeeze())

        # Redimensionnement de -1 à 1
        X_train = X_train / 127.5 - 1.

        if self.channels == 1:
            X_train = np.expand_dims(X_train, axis=3) # S'il y a un seul channel, il faut expand les dimensions de l'ensemble d'entraînement

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        f = open(self.savePath + "data.csv","a+")
        f.write("Time, Epoch, DiscriminatorLoss, DiscriminatorAcc, GeneratorLoss\n")
        f.close()

        debut = time.time()

        for epoch in range(epochs):

            # ----------------------------
            #  Entraînement Discriminateur
            # ----------------------------

            # Choisir une batch aléatoire d'images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Échantillonner le bruit et générer une batch de nouvelles images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Entraîner le discriminateur
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ------------------------
            #  Entraînement Générateur
            # ------------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Entraîner le générateur (pour avoir l'étiquette du discriminateur comme valide)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Progression
            if epoch % 10 == 0:
                delta = time.time() - debut
                f = open(self.savePath + "data.csv","a+")
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
                f.write("%.3f, %d, %f, %.2f%%, %f\n" % (delta, epoch, d_loss[0], 100*d_loss[1], g_loss))
                f.close()

            # Selon l'intervalle de sauvegarde, on sauvegarde les images générées
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    # Fonction qui crée une image d'échantillon
    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Redimensionnement de l'image à 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                if self.channels == 1:
                    axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                else:
                    axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1

        fig.savefig(("{0}{1}.png").format(self.savePath, epoch))
        plt.close()

# Programme principal
if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=100000, batch_size=32, sample_interval=200) # epochs: 30000