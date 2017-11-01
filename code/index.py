import os
import struct
import bson

import pandas as pd

from tqdm import *

data_dir = "../data/"
train_data_dir = "/media/yoni/DATA"

# create a category index file
# Lookup table for categories
categories_path = os.path.join(data_dir, "category_names.csv")
categories_df = pd.read_csv(categories_path, index_col="category_id")

# Maps the category_id to an integer index. This is what we'll use to
# one-hot encode the labels.
categories_df["category_idx"] = pd.Series(
    range(len(categories_df)), index=categories_df.index)

categories_df.to_csv("categories.csv")
print("categories.csv created")


train_bson_path = os.path.join(train_data_dir, "train.bson")
num_train_products = 7069896

# Read the BSON files

# We store the offsets and lengths of all items, allowing us random
# access to the items later.
# Inspired by code from: https://www.kaggle.com/vfdev5/random-item-access
# Note: this takes a few minutes to execute, but we only have to do it
# once (we'll save the table to a CSV file afterwards).


def read_bson(bson_path, num_records, with_categories):
    rows = {}
    with open(bson_path, "rb") as f, tqdm(total=num_records) as pbar:
        offset = 0
        while True:
            item_length_bytes = f.read(4)
            if len(item_length_bytes) == 0:
                break

            length = struct.unpack("<i", item_length_bytes)[0]

            f.seek(offset)
            item_data = f.read(length)
            assert len(item_data) == length

            item = bson.BSON.decode(item_data)
            product_id = item["_id"]
            num_imgs = len(item["imgs"])

            row = [num_imgs, offset, length]
            if with_categories:
                row += [item["category_id"]]
            rows[product_id] = row

            offset += length
            f.seek(offset)
            pbar.update()

    columns = ["num_imgs", "offset", "length"]
    if with_categories:
        columns += ["category_id"]

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "product_id"
    df.columns = columns
    df.sort_index(inplace=True)
    return df


train_offsets_df = read_bson(
    train_bson_path, num_records=num_train_products, with_categories=True)

train_offsets_df.to_csv("train_offsets.csv")
