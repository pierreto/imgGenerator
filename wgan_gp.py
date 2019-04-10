# Pierre To (1734636), Jérémie Jasmin (1800865), Guillaume Vergnolle (1968693)
# INF8225 - Projet
# Base sur : https://github.com/eriklindernoren/Keras-GAN/blob/master/dcgan/dcgan.py
# Base de donnees : https://keras.io/datasets/#cifar10-small-image-classification

from __future__ import print_function, division

from keras.datasets import cifar10
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from functools import partial

import keras.backend as K

import matplotlib.pyplot as plt

import os
import sys

import numpy as np

class RandomWeightedAverage(_Merge):
    """ Mélange de manière aléatoire les images fausses et vraies """
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGANGP():
    def __init__(self):
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        # Fixer les paramètre et l'optimiseur comme recommandé dans le papier
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Création du générateur
        self.generator = self.build_generator()
        self.critic = self.build_critic()

        # Fixer les couches du générateur pendant l'entrainement à la critique
        self.generator.trainable = False

        # Image input (échantillon vrai)
        real_img = Input(shape=self.img_shape)

        # Ajout de bruit
        z_disc = Input(shape=(self.latent_dim,))
        # Création de l'échantillon faux
        fake_img = self.generator(z_disc)

        # Le discriminateur s'entraine sur les échantillons vrais/faux
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Mélange entre échantillon vrai/faux
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Détermine la validité de l'image interpolée
        validity_interpolated = self.critic(interpolated_img)

        # Calcul de la fonction loss
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty'

        self.critic_model = Model(inputs=[real_img, z_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])

        # Fixer les couches de critiques pour le générateur
        self.critic.trainable = False
        self.generator.trainable = True

        # Bruit pour le générateur
        z_gen = Input(shape=(100,))
        # Génération des images à partir du bruit
        img = self.generator(z_gen)
        # Entrainement du discriminateur
        valid = self.critic(img)
        # Définition du modèle du générateur
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Calcul du gradient avec les prédictions réalisées sur les échantillons vrais/faux
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # Norme euclidienne en prenant les carrés ...
        gradients_sqr = K.square(gradients)
        #   ... somme à travers les lignes ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... puis on prend la racine
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # Calculer lambda * (1 - ||grad||)^2 poru chaque exemple
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # Calcul de la moyenne du loss à travers les exemples du batch
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 8 * 8, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((8, 8, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size, sample_interval=50):

        # Chargement de la base de donnée cifar10
        (X_train, y_train), (_, _) = cifar10.load_data()
        X_train = np.array(X_train[np.argwhere(y_train.squeeze() == 5)].squeeze())

        # Redimensionnement de -1 à 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5

        if self.channels == 1:
            X_train = np.expand_dims(X_train, axis=3) # S'il y a un seul channel, il faut expand les dimensions de l'ensemble d'entraînement

        # Vérité terrain
        valid = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1))
        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # Entrainement du Discriminant

                # Selection d'un batch aléatoire
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # Entrainement de la critique
                d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                                [valid, fake, dummy])

            # Entrainement du Générateur

            g_loss = self.generator_model.train_on_batch(noise, valid)

            # Affichage de l'évolution des performances du modèle
            if epoch % 10 == 0:
                print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

            # Sauvegarde des échantillons à la fréquence sample_interval
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Redimensionnement des images 0 - 1
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

        savePath = 'result/w_gan-gp/'
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        
        fig.savefig(("{0}{1}.png").format(savePath, epoch))
        plt.close()


if __name__ == '__main__':
    wgan = WGANGP()
    wgan.train(epochs=30000, batch_size=32, sample_interval=100)
