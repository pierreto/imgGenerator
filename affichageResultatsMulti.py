import numpy
import matplotlib.pyplot as plt

#Fichier à ouvrir (update)
filesName = ['vanilla.csv', 'dcgan.csv', 'sgan.csv', 'lsgan.csv', 'wgan.csv']
styles = ['r-', 'g-', 'b-', 'm-', 'c-']
pasMoyenne = 10
pointsMax = 5000

#[time, epoch, DLoss, DAcc, GLoss]
indices = [[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,5],[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4]]

#Format du subplot (update)
nRow = 2
nCol = 2

multime = []
mulepoch = []
mulDLoss = []
mulDAcc = []
mulGLoss = []

def moyennage(donnees, pas, Emax) :
    ret = []
    for liste in donnees :
        listeT = liste[0:Emax+1]
        ret.append([sum(listeT[i*pas:(i+1)*pas])/pas for i in range(len(listeT)//pas)])
    return ret

#Extraction des données et affichage
for i, fileName in enumerate(filesName) :
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
        time.append(float(lsplit[indices[i][0]]))
        epoch.append(float(lsplit[indices[i][1]]))
        DLoss.append(float(lsplit[indices[i][2]]))
        DAcc.append(float(lsplit[indices[i][3]]))
        GLoss.append(float(lsplit[indices[i][4]]))

    [time, epoch, DLoss, DAcc, GLoss] = moyennage([time, epoch, DLoss, DAcc, GLoss], pasMoyenne, pointsMax) #update

    multime.append(time)
    mulepoch.append(epoch)
    mulDLoss.append(DLoss)
    mulDAcc.append(DAcc)
    mulGLoss.append(GLoss)
    
    f.close()

plt.figure
plt.subplots_adjust(hspace=0.3)
compt = 0

compt += 1
plt.subplot(nRow, nCol, compt)
for ind, loss in enumerate(mulDLoss) :
    plt.plot(mulepoch[ind], loss, styles[ind]) #update
plt.title("Loss Dénominateur")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(['Vanilla', 'DCGAN', 'SGAN', 'LSGAN', 'WGAN'])
plt.grid()

compt += 1
plt.subplot(nRow, nCol, compt)
for ind, accu in enumerate(mulDAcc) :
    plt.plot(mulepoch[ind], accu, styles[ind]) #update
plt.title("Précision Discriminateur")
plt.xlabel("Epoch")
plt.ylabel("Précision (%)")
plt.legend(['Vanilla', 'DCGAN', 'SGAN', 'LSGAN', 'WGAN'])
plt.grid()

compt += 1
plt.subplot(nRow, nCol, compt)
for ind, loss in enumerate(mulGLoss) :
    plt.plot(mulepoch[ind], loss, styles[ind]) #update
plt.title("Loss Générateur")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(['Vanilla', 'DCGAN', 'SGAN', 'LSGAN', 'WGAN'])
plt.grid()

compt += 1
plt.subplot(nRow, nCol, compt)
for ind, timeExec in enumerate(multime) :
    plt.plot(mulepoch[ind], timeExec, styles[ind]) #update
plt.title("Temps d'exécution")
plt.xlabel("Epoch")
plt.ylabel("Temps (s)")
plt.legend(['Vanilla', 'DCGAN', 'SGAN', 'LSGAN', 'WGAN'])
plt.grid()

plt.show()
        
