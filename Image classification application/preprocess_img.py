import tensorflow as tf

image_size = 224

def process_image(test_image):
    # This function takes an image in the form of a Numpy array
    # and returns an image in the form of a Numpy array with shape of (224, 224, 3)
    image_tensor = tf.cast(test_image, tf.float32)
    image_resized = tf.image.resize(image_tensor, (image_size, image_size))
    image_resized /= 255
    final_image = image_resized.numpy().squeeze()
    return final_image