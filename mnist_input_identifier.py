import tensorflow as tf
import numpy as np
from PIL import Image
import argparse

def prep_image(image_file):

    img = Image.open(image_file).convert('L').resize( (28,28) ) # Get image, turn it into grayscale and 28x28 format

    img_array = np.array(img)/255.0

    # Expecting pictures from paint which is black on white, mnist needs white on black
    img_array = 1- img_array # invert colours

    img_array = img_array.reshape(1,28,28)
    
    img_array = tf.cast(img_array, tf.float32)

    return img_array

def main(image_file):

    model = tf.keras.models.load_model('mnist_model.keras')

    prepped_image = prep_image(image_file)

    prediction = model.predict(prepped_image)
    predicted_digit = np.argmax(prediction)

    print(f'The prediction array is: {prediction}')
    print(f'The model thinks you submitted the digit: {predicted_digit}')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST Digit Classifier")
    parser.add_argument('image_file',type=str,help='File path to the image')
    args = parser.parse_args()
    main(args.image_file)

