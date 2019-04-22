# load model
import cv2
import numpy as np
import tensorflow as tf
import sys
import keras
from keras.models import load_model


from data.image_net_classes import image_net_classes


np.set_printoptions(threshold=sys.maxsize)

def load_bagnet_model():
    model_urls = {
        'bagnet9': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/d413271344758455ac086992beb579e256447839/bagnet8.h5',
        'bagnet17': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/d413271344758455ac086992beb579e256447839/bagnet16.h5',
        'bagnet33': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/d413271344758455ac086992beb579e256447839/bagnet32.h5',
    }

    model_path = keras.utils.get_file(
        'bagnet32.h5',
        model_urls['bagnet33'],
        cache_subdir='models',
        file_hash='96d8842eec8b8ce5b3bc6a5f4ff3c8c0278df3722c12bc84408e1487811f8f0f')

    keras_model = load_model(model_path)

    return keras_model


keras_model = load_bagnet_model()

print(keras_model.summary())

# def load_sample_image():
#     image_bgr = cv2.imread("data/example_4.png")

#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

#     resized_image = cv2.resize(image_rgb, (224, 224))

#     channels_first_image = np.rollaxis(resized_image, 2, 0)

#     sample_image = channels_first_image / 255.
#     sample_image -= np.array([0.485, 0.456, 0.406])[:, None, None]
#     sample_image /= np.array([0.229, 0.224, 0.225])[:, None, None]

#     return sample_image


# sample_image = load_sample_image()


# result = keras_model.predict(
#     np.array([sample_image]),
#     batch_size=1,
# )

# winner_index = np.argmax(result)

# winner = image_net_classes[winner_index]

# print("result", result)
# print("prediction: ", winner)
