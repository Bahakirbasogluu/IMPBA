"""
Custom trainer for image inpainting with perceptual loss.
Uses VGG19 features (block4_conv4) for computing perceptual similarity.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


class InpaintingTrainer:
    """Custom trainer for inpainting models using perceptual loss."""
    
    def __init__(self, model, train_data, train_gts, val_data, val_gts, 
                 optimizer, epochs, batch_size, test_input_data, test_index_to_print):
        self.model = model
        self.train_data = train_data
        self.train_gts = train_gts
        self.val_data = val_data
        self.val_gts = val_gts
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.test_input_data = test_input_data
        self.test_index_to_print = test_index_to_print
        
        # Initialize VGG for perceptual loss
        self.vgg = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet', input_shape=(256, 256, 3)
        )
        self.vgg.trainable = False
        self.content_layers = ['block4_conv4']
        self.content_model = tf.keras.Model(
            inputs=self.vgg.input,
            outputs=[self.vgg.get_layer(layer).output for layer in self.content_layers]
        )

    def train(self):
        """Run the training loop."""
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            self._train_epoch()
            self._validate_epoch()
            self._log_predictions(epoch)
            print()
        return self.model

    def _train_epoch(self):
        """Train for one epoch."""
        num_batches = len(self.train_data) // self.batch_size
        total_loss = 0.0

        for batch in range(num_batches):
            batch_start = batch * self.batch_size
            batch_end = (batch + 1) * self.batch_size
            inputs = self.train_data[batch_start:batch_end]
            targets = self.train_gts[batch_start:batch_end]

            with tf.GradientTape() as tape:
                predictions = self.model(inputs, training=True)
                loss = self._compute_perceptual_loss(targets, predictions)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            total_loss += loss.numpy()

        avg_loss = total_loss / num_batches
        print(f"Train Loss: {avg_loss:.4f}")

    def _validate_epoch(self):
        """Validate for one epoch."""
        num_batches = len(self.val_data) // self.batch_size
        total_loss = 0.0

        for batch in range(num_batches):
            batch_start = batch * self.batch_size
            batch_end = (batch + 1) * self.batch_size
            inputs = self.val_data[batch_start:batch_end]
            targets = self.val_gts[batch_start:batch_end]

            predictions = self.model(inputs, training=False)
            loss = self._compute_perceptual_loss(targets, predictions)
            total_loss += loss.numpy()

        avg_loss = total_loss / num_batches
        print(f"Validation Loss: {avg_loss:.4f}")

    def _compute_perceptual_loss(self, y_true, y_pred):
        """Compute perceptual loss using VGG19 features."""
        # Scale images to VGG range [0, 255]
        y_true_vgg = y_true * 255.0
        y_pred_vgg = y_pred * 255.0

        # Get VGG features
        true_features = self.content_model(y_true_vgg)
        pred_features = self.content_model(y_pred_vgg)

        # Compute MSE between features
        loss = 0.0
        for true_f, pred_f in zip(true_features, pred_features):
            loss += tf.reduce_mean(tf.square(true_f / 255.0 - pred_f / 255.0))

        return loss

    def _log_predictions(self, epoch):
        """Log sample predictions during training."""
        if self.test_index_to_print is not None and len(self.test_input_data) > 0:
            test_input = np.expand_dims(self.test_input_data[self.test_index_to_print], axis=0)
            inpainted_image = self.model.predict(test_input, verbose=0)
            inpainted_image = inpainted_image.reshape(inpainted_image.shape[1:])
            print(f'Epoch {epoch+1} - Test Image {self.test_index_to_print + 1} Prediction saved')
