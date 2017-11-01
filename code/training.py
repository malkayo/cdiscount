from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D

from generator import *

model = Sequential()
model.add(Conv2D(32, 3, padding="same",
                 activation="relu", input_shape=(180, 180, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(128, 3, padding="same", activation="relu"))
model.add(MaxPooling2D())
model.add(GlobalAveragePooling2D())
model.add(Dense(num_classes, activation="softmax"))

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

print(model.summary())

# To train the model:
model.fit_generator(
    train_gen,
    steps_per_epoch=10,  # num_train_images // batch_size,
    epochs=3,
    validation_data=val_gen,
    validation_steps=10,  # num_val_images // batch_size,
    workers=8)
