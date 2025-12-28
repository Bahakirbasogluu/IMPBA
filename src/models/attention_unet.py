import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.utils import shuffle

class AttentionUNet(tf.keras.Model):
    def __init__(self):
        super(AttentionUNet, self).__init__()
        # Encoder
        self.conv1 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2))

        self.conv3 = layers.Conv2D(128, 3, padding='same', activation='relu')
        self.conv4 = layers.Conv2D(128, 3, padding='same', activation='relu')
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2))

        self.conv5 = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.conv6 = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.pool3 = layers.MaxPooling2D(pool_size=(2, 2))

        self.conv7 = layers.Conv2D(512, 3, padding='same', activation='relu')
        self.conv8 = layers.Conv2D(512, 3, padding='same', activation='relu')
        self.drop4 = layers.Dropout(0.5)
        self.pool4 = layers.MaxPooling2D(pool_size=(2, 2))

        # Bottleneck
        self.conv9 = layers.Conv2D(1024, 3, padding='same', activation='relu')
        self.conv10 = layers.Conv2D(1024, 3, padding='same', activation='relu')
        self.drop5 = layers.Dropout(0.5)

        # Decoder
        self.upconv6 = layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')
        self.conv11 = layers.Conv2D(512, 3, padding='same', activation='relu')
        self.conv12 = layers.Conv2D(512, 3, padding='same', activation='relu')

        self.upconv7 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')
        self.conv13 = layers.Conv2D(256, 3, padding='same', activation='relu')
        self.conv14 = layers.Conv2D(256, 3, padding='same', activation='relu')

        self.upconv8 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')
        self.conv15 = layers.Conv2D(128, 3, padding='same', activation='relu')
        self.conv16 = layers.Conv2D(128, 3, padding='same', activation='relu')

        self.upconv9 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')
        self.conv17 = layers.Conv2D(64, 3, padding='same', activation='relu')
        self.conv18 = layers.Conv2D(64, 3, padding='same', activation='relu')

        self.conv19 = layers.Conv2D(3, 1, activation='sigmoid')

    def call(self, x):
        # Encoder
        conv1 = self.conv1(x)
        conv1 = self.conv2(conv1)
        pool1 = self.pool1(conv1)

        conv2 = self.conv3(pool1)
        conv2 = self.conv4(conv2)
        pool2 = self.pool2(conv2)

        conv3 = self.conv5(pool2)
        conv3 = self.conv6(conv3)
        pool3 = self.pool3(conv3)

        conv4 = self.conv7(pool3)
        conv4 = self.conv8(conv4)
        drop4 = self.drop4(conv4)
        pool4 = self.pool4(drop4)

        # Bottleneck
        conv5 = self.conv9(pool4)
        conv5 = self.conv10(conv5)
        drop5 = self.drop5(conv5)

        # Decoder
        up6 = self.upconv6(drop5)
        merge6 = tf.concat([drop4, up6], axis=-1)
        conv6 = self.conv11(merge6)
        conv6 = self.conv12(conv6)

        up7 = self.upconv7(conv6)
        merge7 = tf.concat([conv3, up7], axis=-1)
        conv7 = self.conv13(merge7)
        conv7 = self.conv14(conv7)

        up8 = self.upconv8(conv7)
        merge8 = tf.concat([conv2, up8], axis=-1)
        conv8 = self.conv15(merge8)
        conv8 = self.conv16(conv8)

        up9 = self.upconv9(conv8)
        merge9 = tf.concat([conv1, up9], axis=-1)
        conv9 = self.conv17(merge9)
        conv9 = self.conv18(conv9)

        conv10 = self.conv19(conv9)
        return conv10