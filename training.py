import os
import cv2
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

from keras.applications import resnet
from keras.layers import Input
from keras.layers.pooling import AveragePooling2D
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.models import Model

from keras.optimizers import adam_v2

import pickle

datapath = r"D:\mydata"
outputmodel = r"D:\mydata\outputdata\videomodels"
outputlabelbinarize = r"D:\mydata\outputdata\videoclassificationbinarizer"
epoch = 2

sports_label = set(['boxing','table_tennis'])
print("image loading")
pathtoimages = list(paths.list_images(datapath))
data = []
labels = []

for images in pathtoimages:
    label = images.split(os.path.sep)[-2]
    if label not in sports_label:
        continue
    image = cv2.imread(images)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(224,224))
    data.append(image)
    labels.append(label)

data = np.array(data)
labels = np.array(labels)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.25, stratify=labels,
                                                      random_state=42)
traininAugmentation = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)
validationAugmentation = ImageDataGenerator()
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
traininAugmentation.mean = mean
validationAugmentation.mean = mean
print("done")

model = resnet.ResNet50
baseModel = model(weights="imagenet",include_top=False, input_tensor=Input(shape=(224,224,3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(lb.classes_),activation="softmax")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)
for basemodellayers in baseModel.layers:
    basemodellayers.trainable = False

opt = adam_v2.Adam(learning_rate=0.0001, decay=1e-4/epoch)

model.compile(loss="categorical_crossentropy",optimizer=opt, metrics=["accuracy"])

History = model.fit_generator(
    traininAugmentation.flow(x_train, y_train, batch_size=32),
    steps_per_epoch=len(x_train) // 32,
    validation_data=validationAugmentation.flow(x_test, y_test),
    validation_steps=len(x_test) // 32,
    epochs=epoch
)

model.save(outputmodel)
lbinarizer = open(r"D:\mydata\outputdata\videoclassificationbinarizer.pickle","wb")
lbinarizer.write(pickle.dumps(lb))
lbinarizer.close()
