import pandas as pd
import numpy as np

from tqdm import *
from collections import defaultdict

data_dir = "../data/"
train_data_dir = "/media/yoni/DATA"

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

# Create a random train/validation split
# We split on products, not on individual images. Since some of the categories
# only have a few products, we do the split separately for each category.
# This creates two new tables, one for the training images and one for the
# validation images. There is a row for every single image, so if a product has
# more than one image it occurs more than once in the table.


def make_val_set(df, split_percentage=0.2, drop_percentage=0.):
    # Find the product_ids for each category.
    category_dict = defaultdict(list)
    for ir in tqdm(df.itertuples()):
        category_dict[ir[5]].append(ir[0])

    train_list = []
    val_list = []
    with tqdm(total=int((1 - drop_percentage) * len(df))) as pbar:
        for category_id, product_ids in category_dict.items():
            category_idx = cat2idx[category_id]

            # Randomly remove products to make the dataset smaller.
            keep_size = int(len(product_ids) * (1. - drop_percentage))
            if keep_size < len(product_ids):
                product_ids = np.random.choice(
                    product_ids, keep_size, replace=False)

            # Randomly choose the products that become part of
            # the validation set.
            val_size = int(len(product_ids) * split_percentage)
            if val_size > 0:
                val_ids = np.random.choice(
                    product_ids, val_size, replace=False)
            else:
                val_ids = []

            # Create a new row for each image.
            for product_id in product_ids:
                row = [product_id, category_idx]
                for img_idx in range(df.loc[product_id, "num_imgs"]):
                    if product_id in val_ids:
                        val_list.append(row + [img_idx])
                    else:
                        train_list.append(row + [img_idx])
                pbar.update()

    columns = ["product_id", "category_idx", "img_idx"]
    train_df = pd.DataFrame(train_list, columns=columns)
    val_df = pd.DataFrame(val_list, columns=columns)
    return train_df, val_df


split_percentage = 0.2
drop_percentage = 0.8
train_images_df, val_images_df = make_val_set(
    train_offsets_df, split_percentage=split_percentage,
    drop_percentage=drop_percentage)

# debug
# print(train_images_df[:20])
# print(val_images_df[:20])
# debug

print("Number of training images:", len(train_images_df))
print("Number of validation images:", len(val_images_df))
print("Total images:", len(train_images_df) + len(val_images_df))


# Are all categories represented in the train/val split? (Note: if the drop
# percentage is high, then very small categories won't have enough products
# left to make it into the validation set.)

print("Number of categories in (training, validation) set:")
print(len(train_images_df["category_idx"].unique()),
      len(val_images_df["category_idx"].unique()))

# Save the lookup tables as CSV so that we don't need to repeat the above
# procedure again.
train_images_df.to_csv("train_images_split-{}_drop-{}.csv".format(
    split_percentage, drop_percentage))
val_images_df.to_csv("val_images_split-{}_drop-{}.csv".format(
    split_percentage, drop_percentage))
