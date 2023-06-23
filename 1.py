import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical, load_img
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
from tensorflow import keras
print(keras.__version__)
print(os.listdir("./input"))

FAST_RUN = False
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

file_names = os.listdir("./input/train")
categories = []
for file_name in file_names:
    category = file_name.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'file_name': file_names,
    'category': categories
})
#
# print(df.head())
# print(df.tail())

# df['category'].value_counts().plot.bar()
# plt.show()
import torch
print(torch.cuda.is_available())

sample = random.choice(file_names)
image = load_img('./input/train/' + sample)
# print(image)
# plt.imshow(image)
# plt.show()
import tensorflow as tf

print(tf.test.is_gpu_available())

from keras.models import  Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#
# 根据[1]，可知对CNN模型，Param的计算方法如下：
#
# （卷积核长度 * 卷积核宽度通道数+1）*卷积核个数
# 所以，
#
# 第一个CONV层，Conv2D(32, kernel_size=(3, 2), input_shape=(8,8,1))，Param=(3 * 2 * 1+1)*32 = 224.
# 第二个CONV层，Conv2D(64, (2, 3), activation='relu')，经过第一个层32个卷积核的作用，第二层输入数据通道数为32，Param=(2 * 3 * 32+1)*64 = 12352.
# 第三个CONV层，Conv2D(64, (2, 2), activation='relu')，经过第二个层64个卷积核的作用，第二层输入数据通道数为64，Param=(2 * 2 * 64+1)*64 = 16448.
#
# dense_6 (Dense)这里的Param为什么是32896呢？
# 因为经过flatten_4 (Flatten)的作用，输出变为了256，而dense_6 (Dense)中有128个卷积核，所以Param=128*（256+1）= 32896。

print(model.summary())

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
check_point_path="./tmp/check_point"
checkpoint= ModelCheckpoint(check_point_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks = [earlystop, learning_rate_reduction, checkpoint]

df['category'] = df['category'].replace({0: 'cat', 1: 'dog'})
train_df, validate_df = train_test_split(df, test_size=0.2, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
print(train_df.shape)
print(validate_df.shape)

# train_df['category'].value_counts().plot.bar()
# plt.show()

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]

batch_size = 32

train_datagen = ImageDataGenerator(
    rotation_range = 15,
    rescale = 1. / 255,
    shear_range = 0.1,
    zoom_range= 0.2,
    horizontal_flip = True,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
)

validation_datagen  = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    "./input/train/",
    x_col='file_name',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    "./input/train/",
    x_col = 'file_name',
    y_col = 'category',
    target_size = IMAGE_SIZE,
    class_mode = 'categorical',
    batch_size = batch_size,
)

example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df,
    "./input/train/",
    x_col='file_name',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)

# plt.figure(figsize=(12, 12))
# for i in range(0, 15):
#     plt.subplot(5, 3, i+1)
#     for X_batch, Y_batch in example_generator:
#         image = X_batch[0]
#         plt.imshow(image)
#         break
# plt.tight_layout()
# plt.show()


model_name = 'model.h5'

test_filenames = os.listdir("./input/test2/")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df,
    "./input/test2/",
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)

model.load_weights(model_name)

predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))
test_df['category'] = np.argmax(predict, axis=-1)
label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['category'] = test_df['category'].replace(label_map)
test_df['category'] = test_df['category'].replace({ 'dog': 1, 'cat': 0 })

sample_test = test_df.head(20)
sample_test.head()
plt.figure(figsize=(12, 30))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img("./input/test2/"+filename, target_size=IMAGE_SIZE)
    plt.subplot(10, 10, index+1)
    plt.imshow(img)
    plt.xlabel(filename + '(' + "{}".format(category) + ')' )
plt.tight_layout()
plt.show()
#
# tmp_model = "tmp_" + model_name
#
# epochs=3 if FAST_RUN else 50
# history = model.fit(
#     train_generator,
#     epochs = epochs,
#     validation_data= validation_generator,
#     validation_steps = total_validate // batch_size,
#     steps_per_epoch=total_train//batch_size,
#     callbacks = callbacks,
# )
#
# model.save_weights(model_name)
