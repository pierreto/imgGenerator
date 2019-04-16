# Pierre To (1734636), Jérémie Jasmin (1800865), Guillaume Vergnolle (1968693)
# INF8225 - Projet
# Base sur : https://github.com/eriklindernoren/Keras-GAN/blob/master/sgan/sgan.py
# Base de donnees : https://keras.io/datasets/#cifar10-small-image-classification

from __future__ import print_function, division

from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K

import os
import matplotlib.pyplot as plt
import numpy as np
import time

# Implémentation d'un GAN avec apprentissage semi-supervisé avec un generateur et un discriminateur
class SGAN:
    def __init__(self):
        self.img_rows = 32 # Hauteur en pixels des images
        self.img_cols = 32 # Largeur en pixels des images
        self.channels = 3 # Nombre de canaux de couleur
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 10
        self.latent_dim = 100 # Taille du vecteur latent z (taille de l'entrée du générateur)

        self.savePath = 'result/s_gan/'
        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

        optimizer = Adam(0.0002, 0.5) # taux d'aprentissage de 0.0002 et hyperparametre Beta1 pour l'optimisateur Adam

        # Construire le discriminateur
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
            loss=['binary_crossentropy', 'categorical_crossentropy'],
            loss_weights=[0.5, 0.5],
            optimizer=optimizer,
            metrics=['accuracy']
        )

        # Construire le générateur
        self.generator = self.build_generator()

        # Le générateur prend le bruit comme entrée et génère des images
        noise = Input(shape=(100,))
        img = self.generator(noise)

        # Pour ce modèle, on n'entraîne pas le discriminateur (seulement le générateur)
        self.discriminator.trainable = False

        # Le discriminateur prend les images générées comme entrée et détermine la validité
        valid, _ = self.discriminator(img)

        # Modèle combiné (pile de générateur et de générateur)
        # Entraînement du générateur pour tromper le discriminateur
        self.combined = Model(noise, valid)
        self.combined.compile(loss=['binary_crossentropy'], optimizer=optimizer)

    # Fonction qui construit le générateur
    # Fonction d'activation : tanh, ReLU
    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 8 * 8, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((8, 8, 128)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

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

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())

        model.summary()

        # Sauvegarde du discriminateur dans un fichier
        with open(self.savePath + 'discriminator.txt','w+') as fh:
            model.summary(print_fn=lambda x: fh.write(x + '\n'))

        img = Input(shape=self.img_shape)

        features = model(img)
        valid = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes+1, activation="softmax")(features)

        return Model(img, [valid, label])

    # Entraînement
    def train(self, epochs, batch_size=128, sample_interval=50):

        # Charger les données
        (X_train, y_train), (_, _) = cifar10.load_data()
        X_train = np.array(X_train[np.argwhere(y_train.squeeze() == 5)].squeeze())

        # Redimensionnement de -1 à 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        
        if self.channels == 1:
            X_train = np.expand_dims(X_train, axis=3) # S'il y a un seul channel, il faut expand les dimensions de l'ensemble d'entraînement

        y_train = y_train.reshape(-1, 1)
        y_train = np.array(y_train[y_train == 6])

        # Poids des classes :
        # Pour balancer la différence d'occurences entre les étiquettes.
        # 50% des étiquettes que le discriminateur s'entraîne sont "fausses".
        # Weight = 1 / frequency
        half_batch = batch_size // 2
        cw1 = {0: 1, 1: 1}
        cw2 = {i: self.num_classes / half_batch for i in range(self.num_classes)}
        cw2[self.num_classes] = 1 / half_batch

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        f = open(self.savePath + "data.csv","a+")
        f.write("Time, Epoch, DiscriminatorLoss, DiscriminatorAcc, DiscriminatorOpAcc, GeneratorLoss\n")
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

            # Vecteur one-hot pour les étiquettes
            labels = to_categorical(y_train[idx], num_classes=self.num_classes+1)
            fake_labels = to_categorical(np.full((batch_size, 1), self.num_classes), num_classes=self.num_classes+1)

            # Entraîner le discriminateur
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, labels], class_weight=[cw1, cw2])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, fake_labels], class_weight=[cw1, cw2])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ------------------------
            #  Entraînement Générateur
            # ------------------------

            # Entraîner le générateur (pour avoir l'étiquette du discriminateur comme valide)
            g_loss = self.combined.train_on_batch(noise, valid, class_weight=[cw1, cw2])

            # Progression
            if epoch % 10 == 0:
                delta = time.time() - debut
                f = open(self.savePath + "data.csv","a+")
                print ("%d [D loss: %f, acc: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss))
                f.write("%.3f, %d, %f, %.2f, %.2f, %f\n" % (delta, epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss))
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

    # Fonction qui crée une image d'échantillon
    def save_model(self):

        def save(model, model_name):
            model_path = ("{0}{1}.json").format(self.savePath, model_name)
            weights_path = ("{0}{1}_weights.hdf5").format(self.savePath, model_name)
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "cifar10_sgan_generator")
        save(self.discriminator, "cifar10_sgan_discriminator")
        save(self.combined, "cifar10_sgan_adversarial")

# Programme principal
if __name__ == '__main__':
    sgan = SGAN()
    sgan.train(epochs=100000, batch_size=32, sample_interval=200) # epochs : 20000
