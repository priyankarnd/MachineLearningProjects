import argparse
from PIL import Image
import numpy as np
import preprocess_img as pi
import tensorflow as tf
import tensorflow_hub as hub
import json

# Run like this
# python predict.py -i ./test_images/wild_pansy.jpg -m my_model.h5 -c 5 -f label_map.json

# Create a parser
parser = argparse.ArgumentParser(description="Enter the required information along with the flags")

# Add arguments
parser.add_argument("-i", required=True, help="image-path")
parser.add_argument("-m", required=True, help="model")
parser.add_argument("-c", required=True, help="top c number of likely classes")
parser.add_argument("-f", required=True, help="json file name")

# Extract arguments
args = parser.parse_args()

#Print inputs
img_path = args.i
model_name = args.m
classes = int(args.c)
json_file = args.f

with open(json_file, 'r') as f:
    class_names = json.load(f)

'''
print(img_path)
print(model_name)
print(classes)
'''

model = tf.keras.models.load_model(model_name, custom_objects={'KerasLayer':hub.KerasLayer}, compile=False)
#print(model.summary())

image = Image.open(img_path)
np_image = np.asarray(image)
processed_image = pi.process_image(np_image)
input_image = np.expand_dims(processed_image, axis=0)
ps = model.predict(input_image)

probs = np.sort(ps[0])[-classes:]       
probs_list = list(probs)
probs_list.reverse()
# print(probs_list)

label_indices = (-ps[0]).argsort()[:classes]
label_indices = label_indices.tolist()
labels = [class_names[str(i+1)] for i in label_indices]
# print(labels)

print("The top most likely flower names are:",labels, " with the corresponding probabilities of:",probs_list)