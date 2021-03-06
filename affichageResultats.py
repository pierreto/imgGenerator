import numpy
import matplotlib.pyplot as plt

#Fichier à ouvrir (update)
fileName = 'dcgan.csv'
pasMoyenne = 1
pointsMax = 5000

#Format du subplot (update)
nRow = 2
nCol = 2

#Palette de couleur à conserver
[vanilla, dcgan, sgan, lsgan, wgan] = ['r-', 'g-', 'b-', 'm-', 'c-']
style = wgan #update

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
#DOpAcc = []

for line in f:
    lsplit = line.rstrip().rsplit(',')

    #update
    time.append(float(lsplit[0]))
    epoch.append(float(lsplit[1]))
    DLoss.append(float(lsplit[2]))
    DAcc.append(float(lsplit[3]))
    GLoss.append(float(lsplit[4]))
    #DOpAcc.append(float(lsplit[4]))
f.close()

def moyennage(donnees, pas, Emax) :
    ret = []
    for liste in donnees :
        listeT = liste[0:Emax+1]
        ret.append([sum(listeT[i*pas:(i+1)*pas])/pas for i in range(len(listeT)//pas)])
    return ret

[time, epoch, DLoss, DAcc, GLoss] = moyennage([time, epoch, DLoss, DAcc, GLoss], pasMoyenne, pointsMax) #update
#[time, epoch, DLoss, DAcc, GLoss, DOpAcc] = moyennage([time, epoch, DLoss, DAcc, GLoss, DOpAcc], pasMoyenne, pointsMax) #update

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
        
