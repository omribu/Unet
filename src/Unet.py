import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Disable CuDNN to avoid version mismatch
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, Lambda, Dropout, MaxPooling2D, Convolution2DTranspose, concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from pathlib import Path
from tqdm import tqdm
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from PIL import Image


# Check GPU
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU DETECTED: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU detected")


# Check GPU availability
print("CUDA Available:", tf.test.is_built_with_cuda())
print("GPU Available:", tf.test.is_gpu_available())



seed = 42
np.random.seed(seed)


# image size
IMG_HEIGHT = 512
IMG_WIDTH = 512
IMG_CHANNELS = 3


# batch_size = 8 
# lr = 1e-4 ## 0.0001
# ephocs = 100

TRAIN_PATH = '/home/volcani/Unet/data/plowing/train/'
TEST_PATH = '/home/volcani/Unet/data/plowing/test/'
MASK_PATH = '/home/volcani/Unet/data/plowing/mask_train/'

# Train imgaes
train_images = [f for f in os.listdir(TRAIN_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
# Test images
test_images = [f for f in os.listdir(TEST_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Create array with zeros of the same size as the image size (512x512) 
X_train = np.zeros((len(train_images), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_images), IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)

mask_ = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)

for n, id_ in tqdm(enumerate(train_images), total=len(train_images)):
    
    path_img_train = TRAIN_PATH + id_
    path_mask = MASK_PATH + id_

    img_train = imread(path_img_train)[:,:,:IMG_CHANNELS]
    X_train[n] = img_train  # Fill empty X_train with values from img

    # Read the corresponding mask image
    mask_ = imread(path_mask, as_gray=True)  
    mask_ = mask_ / 255.0  
    Y_train[n] = np.expand_dims(mask_, axis=-1)


# Test images
X_test = np.zeros((len(test_images), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),  dtype=np.uint8)
for n, id_ in tqdm(enumerate(test_images), total=len(test_images)):
    path_test_img = TEST_PATH + id_
    img_test = imread(path_test_img)[:, :, :IMG_CHANNELS]
    X_test[n] = img_test 

print('Done!')


# image_x = np.random.randint(0, len(train_images))
# plt.imshow(X_train[image_x])
# plt.show()
# plt.imshow(np.squeeze(Y_train[image_x]))
# plt.show()

###########################################################################

#Build the model
inputs = Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
s = Lambda(lambda x: x / 255)(inputs) # Converting each pixel of the img from integer to float 

# Contraction path
c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = Dropout(0.1)(c1)
c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = MaxPooling2D((2, 2))(c1)


c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = Dropout(0.1)(c2)
c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = MaxPooling2D((2, 2))(c2)

c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = Dropout(0.2)(c3)
c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = MaxPooling2D((2, 2))(c3)

c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = Dropout(0.2)(c4)
c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = MaxPooling2D((2, 2))(c4)

c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = Dropout(0.3)(c5)
c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

# Expansive path
u6 = Conv2DTranspose(128, (2,2), strides=(2, 2), padding='same')(c5)
u6 = concatenate([u6, c4])  # "Skip conections"
c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = Dropout(0.2)(c6)
c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)


u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = concatenate([u7, c3])  # "Skip conections"
c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = Dropout(0.2)(c7)
c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)


u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = concatenate([u8, c2])  # "Skip conections"
c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = Dropout(0.1)(c8)
c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)


u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = concatenate([u9, c1], axis=3) # "Skip conections"
c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = Dropout(0.1)(c9)
c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#########################################
# Model checkpoint
checkpointer = ModelCheckpoint('model_for_plowing.h5', verbose=1, save_best_only=True)

callbacks = [
    checkpointer,
    EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
    TensorBoard(log_dir='/home/volcani/Unet/logs/plowing_experiment')]


result = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=100, callbacks=callbacks)


##################################################

idx = np.random.randint(0, len(X_train))

preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Each pixel is given a value between 0 and 1. We set a threshold .5 to binarize value gets
# threshold predictions to binarize the image.

preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Perform a sanity check on same random training samples
ix = np.random.randint(0, len(preds_train_t))
plt.imshow(X_train[ix])
plt.show()
plt.imshow(np.squeeze(Y_train[ix]))
plt.show()
plt.imshow(np.squeeze(preds_train_t[ix]))
plt.show()


# Perform a sanity check on same random validation samples
ix = np.random.randint(0, len(preds_val_t))
plt.imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
plt.imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
plt.imshow(np.squeeze(preds_val_t[ix]))
plt.show()





