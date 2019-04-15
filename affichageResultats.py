import numpy
import matplotlib.pyplot as plt

#Fichier à ouvrir (update)
fileName = 'data.csv'
pasMoyenne = 10

#Format du subplot (update)
nRow = 2
nCol = 2

#Palette de couleur à conserver
[vanilla, DCGan, SGan, WGan, InfoGan] = ['r-', 'g-', 'b-', 'm-', 'c-']
style = vanilla #update

#Titres, Abscisses et Ordonnées (update)
titles = ['Loss Discriminateur', 'Précision Discriminateur', 'Loss Générateur', 'Temps exécution']
absc = ['Epoch', 'Epoch', 'Epoch', 'Epoch']
ordo = ['Loss', 'Précision (%)', 'Loss', 'Temps (s)']

#Extraction des données et affichage
f = open(fileName, mode='r')
l1 = f.readline().rstrip().rsplit(',')
print(l1)

#update
time = []
epoch = []
DLoss = []
DAcc = []
GLoss = []

for line in f:
    lsplit = line.rstrip().rsplit(',')

    #update
    time.append(float(lsplit[0]))
    epoch.append(float(lsplit[1]))
    DLoss.append(float(lsplit[2]))
    DAcc.append(float(lsplit[3]))
    GLoss.append(float(lsplit[4]))
f.close()

def moyennage(donnees, pas) :
    ret = []
    for liste in donnees :
        ret.append([sum(liste[i*pas:(i+1)*pas])/pas for i in range(len(liste)//pas)])
    return ret

[time, epoch, DLoss, DAcc, GLoss] = moyennage([time, epoch, DLoss, DAcc, GLoss], pasMoyenne) #update

plt.figure
plt.subplots_adjust(hspace=0.3)
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
        
