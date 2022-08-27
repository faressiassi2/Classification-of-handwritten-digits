#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import numpy as np
print(tf.__version__)
#from tensorflow.keras.datasets import mnist
#import PIL.Image, PIL.ImageFont, PIL.ImageDraw


# In[ ]:


#(X_train,y_train),(X_test,y_test) = mnist.load_data()


# In[ ]:


#class myCallback(tf.keras.callbacks.Callback):
#    def on_epoch_end(self, epoch, logs={}):
#        if(logs.get('accuracy')>0.99):
#            print("\nReached 98% accuracy so cancelling training!")
#        self.model.stop_training = True


# In[ ]:


train=pd.read_csv('train0.csv')
test=pd.read_csv('test0.csv')
print("X_Train size :{}\nX_Test size :{}".format(train.shape, test.shape))


# In[ ]:


# Transform Train and Test into images\labels
X_train = train.drop(['label'], axis=1) # all pixel values
y_train = train['label'] # only labels i.e targets digits


# In[ ]:


X_train.shape


# In[ ]:


test.shape


# In[8]:


#en redimensione notre images en 3D car elles sont en origines en 4D voir dataset:(height = 28px, width = 28px , canal = 1)
#L'entrée du réseau est ce que l'on appelle un objet blob.
#Un blob est un objet de tableau numpy 4D (images, canaux, largeur, hauteur).
# 28,28 vient de width, height.
# 1 vient du nombre de canaux.
# -1 signifie que la longueur dans cette dimension est déduite.Ceci est fait sur la base de la contrainte que le nombre 
#d'éléments dans un ndarrayou une Tensorfois remodelé doit rester le même. Dans le didacticiel, chaque image est un vecteur 
#de ligne (784 éléments) et il y a beaucoup de telles lignes (qu'il en soit nainsi, il y a donc des 784néléments).
#ensorflow peut déduire que -1 est n.
#Données en Mnist, entrée n échantillons, chaque échantillon est un vecteur de 784 colonnes. 
#L'entrée est donc une matrice de n * 784. Mais l'entrée du CNN nécessite une convolution,
#et chaque échantillon doit être une matrice.
#La nouvelle forme doit être compatible avec la forme d'origine. S'il s'agit d'un entier,
#le résultat sera un tableau 1-D de cette longueur. Une dimension de forme peut être -1. Dans ce cas, 
#la valeur est déduite de la longueur du tableau et des dimensions restantes .


# In[ ]:


#rendre notre X_train en 4D comme input pour notre reseau convlutional:
#Pour X_train 4D, la forme signifie [taille du lot, hauteur, largeur, canaux].
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)


# In[10]:


X_train.shape


# In[11]:


test.shape


# In[12]:


X_train = X_train / 255
test = test / 255


# In[13]:


#split the train and validation:
#on utilise les données de valid pour évaluer les performances du modèle:
from sklearn.model_selection import train_test_split
train_x, validation_x, train_Y, validation_Y = train_test_split(X_train, y_train, test_size = 0.2)


# In[14]:


train_x.shape


# In[15]:


validation_x.shape


# In[16]:


#le nombre de chaque classe:
print(y_train.value_counts())


# In[17]:


#presque les classes sont equilibré!!!


# In[18]:


print(np.unique(y_train))


# In[19]:


print(np.unique(validation_Y))


# In[20]:


#transformer cette notation ordinal en une notation one-hot-encoding:
from tensorflow.keras import utils
y_label = utils.to_categorical(train_Y,10)
y_test_label = utils.to_categorical(validation_Y,10)


# In[21]:


y_label[0]


# In[22]:


y_label.shape


# In[23]:


#Visualisation de quelques examples des données
plt.figure(figsize=(15,15))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i][:,:,0], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
plt.show()


# In[24]:


model=keras.Sequential([
    keras.layers.Conv2D(32,(3,3),strides=(1,1),padding = 'Same',activation='relu',input_shape = (28,28,1)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(32,(3,3),strides=(1,1),padding = 'Same',activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    #tf.keras.layers.Dropout(0.5),
    keras.layers.Conv2D(64,(3,3),strides=(1,1),padding = 'Same',activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    #tf.keras.layers.Dropout(0.7),
    keras.layers.Conv2D(64,(3,3),strides=(1,1),padding = 'Same',activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    #tf.keras.layers.Dropout(0.7),
    #tf.keras.layers.Conv2D(64,(3,3),padding = 'Same',activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(512,activation='relu'),
    #keras.layers.Dropout(0.9),
    keras.layers.Dense(10,activation='softmax')
])


# In[25]:


model.summary()


# In[36]:


#on utilise Data Augmentation pour éviter l'Overfitting pour cela on utilise la classe ImageDataGenerator:

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# data augmentation
train_datagen = ImageDataGenerator(
      rescale=1/255
      #rotation_range=40,
      #width_shift_range=0.2,
      #height_shift_range=0.2,
      #shear_range=0.2,
      #zoom_range=0.2,
      #horizontal_flip=True,
      #fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1/255)


train_generator = train_datagen.flow(
        X_train,
        y_label
        #batch_size=1000
)
#validation_generator = validation_datagen.flow(
 #       validation_x,
 #       y_test_label,
 #       batch_size=500
#)


# In[26]:


#on utilise ici seulement la metrique accuracy car on utilise la sortie softmax dans notre CNN:
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adamax

model.compile(loss='categorical_crossentropy',
              #loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              #optimizer=SGD(lr=0.01, momentum=0.0),
              #optimizer=Adamax(lr=0.0001),
              #optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])


# In[27]:


#callbacks = myCallback()


# In[28]:


history=model.fit(
       train_x, 
       y_label, 
       batch_size=64,
       epochs=5,
       validation_data=(validation_x,y_test_label)
)


# In[29]:


#Evaluate the model on the test data using `evaluate`: 
score = model.evaluate(validation_x,y_test_label, verbose=0)
print("test loss, test acc:", score)


# In[30]:


# Plot the loss and accuracy curves for training and validation:

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs,acc,label='Training accuracy')
plt.plot(epochs,val_acc,label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(epochs,loss,label='Training Loss')
plt.plot(epochs,val_loss,label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


# In[32]:


#on évalue notre modele on utilisant cross_validation:
#from sklearn.model_selection import cross_val_score
#score = cross_val_score(model,train_generator,scoring='accuracy',cv=4)
#on ne peut pas utiliser cross_valaidation dans le deep learning ??!!!!


# In[31]:


#predict the result:
#Comme les prévisions que vous obtenez sont des valeurs à virgule flottante, 
#il ne sera pas possible de comparer les étiquettes prédites avec les étiquettes de test réelles.
ypred=model.predict_classes(validation_x)


# In[32]:


ypred[0]


# In[33]:


#notre validation labels sont en mode one_hot_encoding donc on doit les transformer avec argmax en :
yy = np.argmax(y_test_label, axis=1)


# In[34]:


yy[0]


# In[35]:


y_test_label[0]


# In[36]:


yy[0]


# In[37]:


from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix


# In[38]:


confusion_matrix(ypred, yy)


# In[39]:


recall_score(ypred, yy, average='micro')


# In[40]:


precision_score(ypred, yy, average='micro')


# In[41]:


f1_score(ypred, yy, average='micro')


# In[42]:


# predict_classes() method :


# In[80]:


ypred2 = model.predict_classes(test)


# In[81]:


#ypred2


# In[82]:


ypred2[0]


# In[ ]:


# predict() method :


# In[75]:


ypred3 = model.predict(test)


# In[76]:


ypred3[0]


# In[80]:


uo = np.round(ypred3)


# In[82]:


uo[0]


# In[84]:


arg = np.argmax(uo, axis=1)


# In[85]:


arg[0]


# In[96]:


test[0].reshape(28,28)


# In[78]:


ypred2[:9]


# In[88]:


# on va juste tracer les prédictions faite par notre modèle sur le dernier dataset qui est le test set : 
#correct = np.where(ypred2)[0]
#correct2 = ypred2
print(ypred2)
#print(correct2)
#print (len(correct2))
for i, j in enumerate(ypred2[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(test[i].reshape(28,28), cmap='gray', interpolation='none')
    #print(i)
    #print(j)
    plt.title("Predicted {}".format(j))
    plt.tight_layout()


# In[74]:


xx = ypred2[1]
plt.subplot(3,3,1)
plt.imshow(test[1].reshape(28,28), cmap='gray', interpolation='none')
plt.title("Predicted {}".format(xx))
plt.tight_layout()


# In[ ]:


#Sauvegardons le modèle afin que vous puissiez le charger directement sans avoir à le réentraîner pendant 20 époques. 
#De cette façon, vous pourrez charger le modèle ultérieurement si vous en avez besoin et modifier l'architecture. 
#Vous pouvez également démarrer le processus de formation sur ce modèle enregistré. 
#C'est toujours une bonne idée de sauvegarder le modèle - et même les poids du modèle! - 
#car cela vous fait gagner du temps. Notez que vous pouvez également enregistrer le modèle après chaque époque.
#Ainsi, en cas de problème qui interrompt l’entraînement à une époque donnée, vous ne devez pas recommencer l’entraînement 
#à partir du début.
#model.save("digit_model.h5py")


# In[ ]:


correct = np.where(predicted_classes==test_Y)[0]
print ("Found %d correct labels" % len(correct))
for i, correct in enumerate(correct[:9]):
plt.subplot(3,3,i+1)
plt.imshow(test_X[correct].reshape(28,28), cmap='gray', interpolation='none')
plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
plt.tight_layout()


# In[90]:


import numpy as np
ypred = np.argmax(ypred,axis = 1)
ypred = pd.Series(ypred,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),ypred],axis = 1)
submission.to_csv("SUB.csv",index=False)


# In[91]:


submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),ypred],axis = 1)
submission.to_csv("SUB5.csv",index=False)


# In[ ]:


#regler notre hyperparametre pour obtenir des meilleurs performances pour notre modele:
       #1-ajouter ou diminuer les couches de notre modele.
       #2-changer optimizer pour notre modele.


# In[1]:


#Pour une image 4D, la forme signifie [taille du lot, hauteur, largeur, canaux]. Ici puisque batchsize = -1, 
#cela signifie que je ne le spécifie pas auparavant.
#Définir une dimension sur -1 (ou Aucun) indique que vous ne vous souciez pas du nombre qui va y entrer. 
#Après avoir construit votre réseau, vous pouvez même voir (en les imprimant) que -1 (ou Aucun) va également 
#apparaître dans d'autres dimensions de couche.

