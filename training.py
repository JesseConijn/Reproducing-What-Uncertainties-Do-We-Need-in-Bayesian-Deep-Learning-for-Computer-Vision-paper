import os
from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow as tf
from PIL import Image
from tensorflow.python.client import device_lib

def save_model(model, path_to_save):
    return model.save(path_to_save)

def load_model(model, path_to_save):
    model = tf.keras.models.load_model(path_to_save)
    return model

def compute_loss(logits, labels):
    # Compute loss using appropriate loss function
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

def normalize_images(images):
    images = images / 255.
    images -= 0.4
    images /= 0.3
    return images


def train():
    # Load tensors
    images_path = './CamVid/'
    output_path_processing = './camvid-preprocessed'
    image_height2       = 360
    image_width2        = 480
    image_height1       = 224
    image_width1        = 224
    model_save_path     = './densenet_model.h5'
    
    batch_size_224x224  = 1
    batch_size_360x480  = 4

    num_epochs          = 100
    learning_rate       = 0.001
    weight_decay        = 1e-4
    
    train_images1, test_images1, val_images1, train_labels1, test_labels1, val_labels1 = preprocess(images_path, output_path_processing, image_height1, image_width1)
    train_images2, test_images2, val_images2, train_labels2, test_labels2, val_labels2 = preprocess(images_path, output_path_processing, image_height2, image_width2)
    
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate) 

    # Train the DenseNet model
    model = densenet()

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_images1, train_labels1, batch_size=batch_size_224x224, epochs=num_epochs, validation_data=(test_images1, test_labels1))

    print(history.history['loss'])     # Training loss for each epoch
    print(history.history['val_loss']) # Validation loss for each epoch
    print(history.history['accuracy']) # Training accuracy for each epoch
    print(history.history['val_accuracy']) # Validation accuracy for each epoch

    eval_loss, eval_accuracy = model.evaluate(val_images1, val_labels1)
    save_model(model, model_save_path)

    # Evaluate the model
    print("Evaluation loss:", eval_loss)
    print("Evaluation accuracy:", eval_accuracy)
    
    # Train the DenseNet model on the second batch of data
    history = model.fit(train_images2, train_labels2, batch_size=batch_size_360x480, epochs=num_epochs, validation_data=(test_images2, test_labels2))

    # Print training history for the second batch of data
    print(history.history['loss'])     # Training loss for each epoch
    print(history.history['val_loss']) # Validation loss for each epoch
    print(history.history['accuracy']) # Training accuracy for each epoch
    print(history.history['val_accuracy']) # Validation accuracy for each epoch

    # Evaluate the model on the second batch of data
    eval_loss, eval_accuracy = model.evaluate(val_images2, val_labels2)
    print("Evaluation loss on the second batch:", eval_loss)
    print("Evaluation accuracy on the second batch:", eval_accuracy)

    # Optionally, save the model
    save_model(model, model_save_path)

    print("Done: Training is completed!")
        
if __name__ == '__main__':
    
    train()
