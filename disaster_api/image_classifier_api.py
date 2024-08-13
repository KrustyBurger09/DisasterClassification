import tensorflow as tf
import PIL
from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from PIL import UnidentifiedImageError
from PIL import ImageFile  
import numpy as np
import cv2
import matplotlib.pyplot as plt

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

## function to classify image
def image_classify(data):
    model1=tf.keras.models.load_model("static/image_classifier.h5")
    model1.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    predictions = []
    img=tf.keras.preprocessing.image.load_img(data)
    img= tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.preprocessing.image.smart_resize(img, (256,256))
    img = tf.reshape(img, (-1, 256,256, 3))

    labels = ['Damaged_Infrastructure', 'Fire_Disaster', 'Human_Damage', 'Land_Disaster', 'Non_Damage']

    prediction = model1.predict(img/255)

    predicted_class_indices=np.argmax(prediction,axis=1)
    
    predictions = [labels[k] for k in predicted_class_indices]
    print(predictions[0])
    return predictions[0]

if __name__ == "__main__":
    data = str(input("enter file name - "))
    print(image_classify(data))