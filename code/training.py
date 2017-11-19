from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
# from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D

from keras.callbacks import TensorBoard
from generator import *

model = Sequential()
model.add(Conv2D(32, 3, padding="same",
                 activation="relu", input_shape=(180, 180, 3)))
# model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(MaxPooling2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
# model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(MaxPooling2D())

model.add(Conv2D(128, 3, padding="same", activation="relu"))
# model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(MaxPooling2D())
model.add(GlobalAveragePooling2D())
model.add(Dense(num_classes, activation="softmax"))

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

print(model.summary())

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=False)

# To train the model:
model.fit_generator(
    train_gen,
    steps_per_epoch=100,  # num_train_images // batch_size,
    epochs=1000,
    validation_data=val_gen,
    validation_steps=1000,  # num_val_images // batch_size,
    workers=16,
    callbacks=[tensorboard])


def save_model(model, modelf_location):
    # Serialize model architecture to JSON
    model_json = model.to_json()
    with open(modelf_location + ".json", "w") as json_file:
        json_file.write(model_json)

    # Save model weights
    model.save_weights(modelf_location + ".h5", overwrite=True)


save_model(model, "./18112017model")

# around 23%
