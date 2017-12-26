from tqdm import *
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array

import pandas as pd
import numpy as np

import os
import io
import bson

num_test_products = 1768182


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


model = load_model("./28112017modelplus")

categories_df = pd.read_csv("categories.csv", index_col="category_id")
train_offsets_df = pd.read_csv("train_offsets.csv")


def make_category_tables():
    # Create dictionaries for quick lookup of `category_id` to
    # `category_idx` mapping.
    cat2idx = {}
    idx2cat = {}
    for ir in categories_df.itertuples():
        category_id = ir[0]
        category_idx = ir[4]
        cat2idx[category_id] = category_idx
        idx2cat[category_idx] = category_id
    return cat2idx, idx2cat

cat2idx, idx2cat = make_category_tables()
train_data_dir = "../data/"
submission_df = pd.read_csv(train_data_dir + "sample_submission.csv")
print("submission_df:")
print(submission_df.head())

# test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator()

test_bson_path = "../data/test.bson"
data = bson.decode_file_iter(open(test_bson_path, "rb"))

with tqdm(total=num_test_products) as pbar:
    for c, d in enumerate(data):
        product_id = d["_id"]
        num_imgs = len(d["imgs"])

        batch_x = np.zeros((num_imgs, 180, 180, 3), dtype=K.floatx())

        for i in range(num_imgs):
            bson_img = d["imgs"][i]["picture"]

            # Load and preprocess the image.
            img = load_img(io.BytesIO(bson_img), target_size=(180, 180))
            x = img_to_array(img)
            x = test_datagen.random_transform(x)
            x = test_datagen.standardize(x)

            # Add the image to the batch.
            batch_x[i] = x

        prediction = model.predict(batch_x, batch_size=num_imgs)
        avg_pred = prediction.mean(axis=0)
        cat_idx = np.argmax(avg_pred)

        submission_df.iloc[c]["category_id"] = idx2cat[cat_idx]
        pbar.update()

submission_df.to_csv("20172911_submission.csv.gz", compression="gzip", index=False)
