import numpy
import matplotlib.pyplot as plt
import csv

#Fichier à ouvrir
fileName = 'data.csv'

#Format du subplot
nRow = 1
nCol = 4

#Palette de couleur à conserver
palette = ['r-', 'g-', 'b-', 'm-', 'c-'] # Vanilla:0, D-Gan:1, S-Gan:2, W-Gan:3, Info-Gan:4
style = palette[4]

#Titres, Abscisses et Ordonnées
titles = ['Loss Discriminateur', 'Précision Discriminateur', 'Loss Générateur', 'Temps exécution']
absc = ['Epoch', 'Epoch', 'Epoch', 'Epoch']
ordo = ['Loss', 'Précision', 'Loss', 'Temps (s)']

f = open(fileName, mode='r')
l1 = f.readline().rstrip().rsplit(',')
print(l1)

time = []
epoch = []
DLoss = []
DAcc = []
GLoss = []

for line in f:
    lsplit = line.rstrip().rsplit(',')
    time.append(float(lsplit[0]))
    epoch.append(float(lsplit[1]))
    DLoss.append(float(lsplit[2]))
    DAcc.append(float(lsplit[3]))
    GLoss.append(float(lsplit[4]))

plt.figure
compt = 0

compt += 1
plt.subplot(nRow, nCol, compt)
plt.plot(epoch, DLoss, style) #update
plt.title(titles[compt-1])
plt.xlabel(absc[compt-1])
plt.ylabel(ordo[compt-1])
plt.grid()

compt += 1
plt.subplot(nRow, nCol, compt)
plt.plot(epoch, DAcc, style) #update
plt.title(titles[compt-1])
plt.xlabel(absc[compt-1])
plt.ylabel(ordo[compt-1])
plt.grid()

compt += 1
plt.subplot(nRow, nCol, compt)
plt.plot(epoch, GLoss, style) #update
plt.title(titles[compt-1])
plt.xlabel(absc[compt-1])
plt.ylabel(ordo[compt-1])
plt.grid()

compt += 1
plt.subplot(nRow, nCol, compt)
plt.plot(epoch, time, style) #update
plt.title(titles[compt-1])
plt.xlabel(absc[compt-1])
plt.ylabel(ordo[compt-1])
plt.grid()

plt.show()
