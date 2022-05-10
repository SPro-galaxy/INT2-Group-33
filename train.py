import os
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.regularizers import l2

model_name = "model"

def load_dataset():
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    
    # One-hot encode target values.
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


def prep_pixels(train, test):
    # Convert from ints to floats and normalise (0-1).
    train_norm = train.astype('float32') / 255.0
    test_norm = test.astype('float32') / 255.0
    return train_norm, test_norm


def make_model():
    model = Sequential([
        Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)),
        BatchNormalization(),
        
        Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        
        Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        BatchNormalization(),
        
        Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.5),
        
        Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        BatchNormalization(),
        
        Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.55),
        
        Flatten(),
        Dense(772, activation='relu', kernel_initializer='he_uniform'),
        BatchNormalization(),
        Dropout(0.65),

        Dense(772, activation='relu', kernel_initializer='he_uniform'),
        BatchNormalization(),
        Dropout(0.75),
        
        Dense(10, activation='softmax')
    ])
    
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

 
if __name__ == "__main__":
    trainX, trainY, testX, testY = load_dataset()

    trainX, testX = prep_pixels(trainX, testX)

    model = make_model()

    # Create data generator for data augmentation.
    datagen = ImageDataGenerator(
        width_shift_range=0.1, 
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    
    # Some logging settings.
    logdir = os.path.join("logs", model_name)
    tensorboard = TensorBoard(log_dir=logdir)
    
    # Train the model.
    it_train = datagen.flow(trainX, trainY, batch_size=32)
    steps = int(trainX.shape[0] / 32)
    model.fit(
        it_train,
        steps_per_epoch=steps, 
        epochs=1600, 
        validation_data=(testX, testY), 
        verbose=1, 
        callbacks=[tensorboard]
    )
    
    # Evaluate.
    _, acc = model.evaluate(testX, testY, verbose=0)
    print('Final val_accuracy: %.3f' % (acc * 100.0))

    model.save(f'{model_name}.h5')
