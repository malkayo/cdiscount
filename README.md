# cdiscount

Code used for the Cdiscount’s Image Classification Challenge on [Kaggle](https://www.kaggle.com/c/cdiscount-image-classification-challenge/)

A modest score of 0.58 was obtained by retraining InceptionV3 using keras.

The training set was restricted to the product classes having not being in the 10% least frequent in the original training set (see "Category distribution" notebook for rationale).

I largely borrowed code from the kaggle kernel: https://www.kaggle.com/humananalog/keras-generator-for-reading-directly-from-bson
  
