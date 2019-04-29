import imageio
import matplotlib.pyplot as plt

duration = 1
pas = 200
bornes = [2000, 10000]
sauts = [1, 5, 10]
xlab = 'S-GAN'
#filenames = [str(200*i)+'.png' for i in range(100)]
#filenames = ['50000.png']

def generateFilenames(pas, bornes, sauts) :
    compteur = -pas
    noms = []
    while compteur < 50000 :
        if compteur < bornes[0] :
            saut = sauts[0]
        elif compteur < bornes[1] :
            saut = sauts[1]
        else :
            saut = sauts[2]
        compteur += pas*saut
        noms.append(str(compteur)+'.png')
    return noms
        
def generateLabels(filenames, pas, bornes, sauts, xlab) :
    for filename in filenames :
        plt.figure
        f = plt.imread(filename)
        compteur = int(filename[:-4])
        if compteur < bornes[0] :
            saut = sauts[0]
        elif compteur < bornes[1] :
            saut = sauts[1]
        else :
            saut = sauts[2]
        plt.imshow(f)
        plt.axis('off')
        plt.title(xlab+', Epoch n° : ' + filename[:-4] + ', Pas : ' + str(pas*saut))
        plt.xlabel(xlab)
        #plt.show()
        plt.savefig('lab'+filename)
         

def generateGif(filenames, extension, timePerImage) :
    with imageio.get_writer('result.gif', mode='I', duration=timePerImage) as writer:
        for filename in filenames:
            image = imageio.imread(extension+filename)
            writer.append_data(image)
            
print("Génération des noms de fichier...")
filenames = generateFilenames(pas, bornes, sauts)
print("Génération des labels...")
generateLabels(filenames, pas, bornes, sauts, xlab)
print("Génération du GIF...")
generateGif(filenames, 'lab', duration)
