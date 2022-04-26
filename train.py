from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow_datasets as tfds
import os

version = "0.1.0tc9"
epochs = 90

# Hyper-parameters
batch_size = 64

# 10 categories of images in (CIFAR-10)
num_classes = 10

def create_model(input_shape):
    """
    Constructs the model:
        - 32 Convolutional (3x3)
        - Relu
        - 32 Convolutional (3x3)
        - Relu
        - Max pooling (2x2)
        - Dropout

        - 64 Convolutional (3x3)
        - Relu
        - 64 Convolutional (3x3)
        - Relu
        - Max pooling (2x2)
        - Dropout

        - 128 Convolutional (3x3)
        - Relu
        - 128 Convolutional (3x3)
        - Relu
        - Max pooling (2x2)
        - Dropout
        
        - Flatten (To make a 1D vector out of convolutional layers)
        - 1024 Fully connected units
        - Relu
        - Dropout
        - 10 Fully connected units (each corresponds to a label category (cat, dog, etc.))
    """

    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))

    # Flattening the convolutions.
    model.add(Flatten())

    # Fully-connected layers.
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(Dropout(0.1))

    model.add(Dense(num_classes, activation="softmax"))

    # Print the summary of the model architecture.
    model.summary()

    # Training the model using adam optimizer.
    opt = Adam(learning_rate=0.0005)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


def load_data():
    """
    This function loads CIFAR-10 dataset, and preprocess it
    """
    
    def preprocess_image(image, label):
        # Convert [0, 255] range integers to [0, 1] range floats.
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image, label
    
    # Loading the CIFAR-10 dataset, splitted between train and test sets.
    ds_train, info = tfds.load("cifar10", with_info=True, split="train", as_supervised=True)
    ds_test = tfds.load("cifar10", split="test", as_supervised=True)
    
    # Repeat dataset forever, shuffle, preprocess, split by batch.
    ds_train = ds_train.repeat().shuffle(1024).map(preprocess_image).batch(batch_size)
    ds_test = ds_test.repeat().shuffle(1024).map(preprocess_image).batch(batch_size)
    return ds_train, ds_test, info


if __name__ == "__main__":
    ds_train, ds_test, info = load_data()

    # Constructs the model
    model = create_model(input_shape=info.features["image"].shape)

    # Logging
    logdir = os.path.join("logs", f"cifar10-model-{version}")
    tensorboard = TensorBoard(log_dir=logdir)

    # Make sure results folder exist.
    if not os.path.isdir("results"):
        os.mkdir("results")

    # Train the model.
    model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=1,
              steps_per_epoch=info.splits["train"].num_examples // batch_size,
              validation_steps=info.splits["test"].num_examples // batch_size,
              callbacks=[tensorboard])

    # Save the model to disk.
    model.save(f"results/cifar10-model-{version}.h5")