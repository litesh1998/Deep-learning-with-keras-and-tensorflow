# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf
import numpy as np


# %%
print(tf.__version__)


# %%
(x_train, y_train), (x_test, y_test)=  tf.keras.datasets.mnist.load_data()


# %%
num_lables=len(np.unique(y_train))
y_train=tf.keras.utils.to_categorical(y_train)
y_test=tf.keras.utils.to_categorical(y_test)


# %%
import os
os.getcwd()


# %%
image_size=x_train.shape[1]
x_train = np.reshape(x_train,[-1, image_size, image_size, 1])
x_test = np.reshape(x_test,[-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


# %%
input_shape = (image_size, image_size, 1)
batch_size = 32
kernel_size = 3
dropout = 0.4
n_filters = 32


# %%
left_inputs = tf.keras.layers.Input(shape=input_shape)
filters=n_filters
x = left_inputs
for i in range(3):
    x = tf.keras.layers.Conv2D(filters=filters,
    kernel_size=kernel_size,
    padding='same',
    activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    filters *= 2


# %%
right_inputs = tf.keras.layers.Input(shape=input_shape)
filters=n_filters
y = left_inputs
for i in range(3):
    y = tf.keras.layers.Conv2D(filters=filters,
    kernel_size=kernel_size,
    padding='same',
    activation='relu',
    dilation_rate=2)(y)
    y = tf.keras.layers.Dropout(dropout)(y)
    y = tf.keras.layers.MaxPooling2D()(y)
    filters *= 2


# %%
y=tf.keras.layers.concatenate([x, y])
y=tf.keras.layers.Flatten()(y)
y=tf.keras.layers.Dropout(dropout)(y)
output=tf.keras.layers.Dense(num_lables, activation='softmax')(y)
model=tf.keras.models.Model([left_inputs, right_inputs], output)


# %%
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# %%
model.summary()


# %%
tf.keras.utils.plot_model(model, to_file='Y-cnn-model.png', show_shapes=True)


# %%
model.fit([x_train, x_train],
y_train,
validation_data=([x_test, x_test], y_test),
epochs=20,
batch_size=batch_size)


# %%
score = model.evaluate([x_test, x_test],
y_test,
batch_size=batch_size,
verbose=1)


# %%


