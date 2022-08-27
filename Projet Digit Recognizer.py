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

#(X_train,y_train),(X_test,y_test) = mnist.load_data()

#class myCallback(tf.keras.callbacks.Callback):
#    def on_epoch_end(self, epoch, logs={}):
#        if(logs.get('accuracy')>0.99):
#            print("\nReached 98% accuracy so cancelling training!")
#        self.model.stop_training = True

#read the CSV dataset
train=pd.read_csv('train0.csv')
test=pd.read_csv('test0.csv')
print("X_Train size :{}\nX_Test size :{}".format(train.shape, test.shape))

# Transform Train and Test into images\labels
X_train = train.drop(['label'], axis=1) # all pixel values
y_train = train['label'] # only labels i.e targets digits
X_train.shape
test.shape

#rendre notre X_train en 4D comme input pour notre reseau convlutional:
#Pour X_train 4D, la forme signifie [taille du lot, hauteur, largeur, canaux].
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

# we normalize our images:
X_train = X_train / 255
test = test / 255

#split the train and validation:
#on utilise les données de valid pour évaluer les performances du modèle:
from sklearn.model_selection import train_test_split
train_x, validation_x, train_Y, validation_Y = train_test_split(X_train, y_train, test_size = 0.2)

#le nombre de chaque classe:
print(y_train.value_counts())

#transformer cette notation ordinal en une notation one-hot-encoding:
from tensorflow.keras import utils
y_label = utils.to_categorical(train_Y,10)
y_test_label = utils.to_categorical(validation_Y,10)

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

model.summary()

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

history=model.fit(
       train_x, 
       y_label, 
       batch_size=64,
       epochs=100,
       validation_data=(validation_x,y_test_label)
)


#Evaluate the model on the test data using `evaluate`: 
score = model.evaluate(validation_x,y_test_label, verbose=0)
print("test loss, test acc:", score)

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



#predict the result:
#Comme les prévisions que vous obtenez sont des valeurs à virgule flottante, 
#il ne sera pas possible de comparer les étiquettes prédites avec les étiquettes de test réelles.
ypred=model.predict_classes(validation_x)

#notre validation labels sont en mode one_hot_encoding donc on doit les transformer avec argmax en :
yy = np.argmax(y_test_label, axis=1)


from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
confusion_matrix(ypred, yy)
recall_score(ypred, yy, average='micro')
precision_score(ypred, yy, average='micro')
f1_score(ypred, yy, average='micro')
