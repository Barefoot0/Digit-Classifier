import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse
import os


datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2
)

def create_model(x_train, y_train):
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten()) # turn 2D matrix into 1D vector (needed because dense layers expect 1D input)
    model.add(tf.keras.layers.Dense(256, activation="relu")) # 1st layer with 256 neurons, using ReLU: f(x) = max(0,x)
    model.add(tf.keras.layers.Dense(128, activation="relu")) # 2nd later with 128 neurons, same activation function as above
    model.add(tf.keras.layers.Dense(10, activation="softmax")) # 10 neurons becase there are 10 possible outputs (0-9)

    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    return model

def main():  
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrain', action='store_true', help='Flag to retrain the model')
    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    x_train, x_test = tf.cast(x_train/255.0, tf.float32), tf.cast(x_test/255.0, tf.float32) # images become normalized 32-bit floats
    y_train, y_test = tf.cast(y_train,tf.int64), tf.cast(y_test,tf.int64) # labels become 64-bit integers

    x_train = tf.expand_dims(x_train, axis=-1)
    x_test = tf.expand_dims(x_test, axis=-1)


    history = None

    model_file = 'mnist_model.keras'
    if args.retrain or not os.path.exists(model_file):
        model = create_model(x_train, y_train)

        history = model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=10, validation_data=(x_test, y_test))
        model.save(model_file) # good but should save it using newer keras method
    else:
        model = load_model(model_file)

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Test Loss: {loss}')
    print(f'Test Accuracy: {accuracy}')
  
if __name__ == '__main__':
    main()
