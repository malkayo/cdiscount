from keras.models import Sequential
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import Adam
# from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D

from keras.callbacks import TensorBoard
from generator import *
from keras.applications import InceptionV3
import os
# base_model = InceptionV3(weights='imagenet', include_top=False)
FC_SIZE = 2048

# def add_new_last_layer(base_model, nb_classes):
#     """Add last layer to the convnet
#     Args:
#       base_model: keras model excluding top
#       nb_classes: # of classes
#     Returns:
#       new keras model with last layer
#     """
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(FC_SIZE, activation='relu')(x)
#     predictions = Dense(nb_classes, activation='softmax')(x)
#     model = Model(input=base_model.input, output=predictions)
#     return model


# model = add_new_last_layer(base_model, nb_classes=num_classes)
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


def load_model(modelf_location):
    from keras.models import model_from_json

    # Load model architecture from JSON
    model_file = modelf_location + ".json"
    if os.path.isfile(model_file):
        json_file = open(model_file, 'r')
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
    else:
        print("Error - Policy model loading - {} missing".format(
            model_file))
        return None

    # Load weights from h5 file
    weights_file = modelf_location + ".h5"
    if os.path.isfile(weights_file):
        print("Loading weights from {}".format(weights_file))
        model.load_weights(weights_file)
    else:
        print("Error - Policy weight loading - {} missing".format(
            weights_file))
        return None

    return model


model = load_model("./19112017model")
model.compile(optimizer=adam,
              loss="categorical_crossentropy",
              metrics=["accuracy"])

print(model.summary())

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=False)

# To train the model:
model.fit_generator(
    train_gen,
    steps_per_epoch=100,  # num_train_images // batch_size,
    epochs=300,
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


save_model(model, "./19112017model")

# around 23%
