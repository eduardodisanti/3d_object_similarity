import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np

def get_tf_embedding(loaded_model, pc_array):
    """
    Processes a point cloud with a TensorFlow model and returns the embedding.

    Args:
        loaded_model (object): A model loaded using tf.saved_model.load().
        pc_array (np.ndarray): The point cloud as a NumPy array of shape (N, 3).
    
    Returns:
        np.array: The embedding vector of the 3D model.
    """
    # 1. Convert to TensorFlow tensor and adjust shape
    pc_tensor = tf.convert_to_tensor(pc_array, dtype=tf.float32)
    pc_tensor = tf.expand_dims(pc_tensor, axis=0) # Add the batch dimension
    
    # 2. Get the embedding
    # The output from a SavedModel can be a dictionary
    model_output = loaded_model(pc_tensor)

    # It's highly likely that the embedding is under a specific key,
    # and the classification output is under another key.
    # The key names depend on how the model was saved.
    
    # Try to find the correct key for the embedding
    # Example keys could be 'embedding', 'features', 'dense_0', etc.
    if isinstance(model_output, dict):
        # You need to print the keys to find the correct one
        embedding_vector = model_output['your_embedding_key'] # Change this key
    else:
        # If the output is a single tensor, it might be the embedding itself.
        embedding_vector = model_output
    
    return embedding_vector.numpy().flatten()