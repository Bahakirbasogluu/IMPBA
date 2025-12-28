"""
U-Net Like model for image inpainting.
Traditional encoder-decoder architecture with skip connections.
"""

from tensorflow import keras


class UNetLikeModel:
    """Constructs a U-Net like model for image inpainting."""

    def __init__(self, input_shape=(256, 256, 4)):
        inputs = keras.layers.Input(input_shape)

        enc1, down1 = self._encoder_block(64, (3, 3), (2, 2), 'relu', 'same', inputs)
        enc2, down2 = self._encoder_block(128, (3, 3), (2, 2), 'relu', 'same', down1)
        enc3, down3 = self._encoder_block(256, (3, 3), (2, 2), 'relu', 'same', down2)
        enc4, down4 = self._encoder_block(512, (3, 3), (2, 2), 'relu', 'same', down3)
        enc5, down5 = self._encoder_block(1024, (3, 3), (2, 2), 'relu', 'same', down4)

        dec1, up1 = self._decoder_block(512, 512, (3, 3), (2, 2), (2, 2), 'relu', 'same', down5, enc5)
        dec2, up2 = self._decoder_block(256, 256, (3, 3), (2, 2), (2, 2), 'relu', 'same', up1, enc4)
        dec3, up3 = self._decoder_block(128, 128, (3, 3), (2, 2), (2, 2), 'relu', 'same', up2, enc3)
        dec4, up4 = self._decoder_block(64, 64, (3, 3), (2, 2), (2, 2), 'relu', 'same', up3, enc2)
        final_conv = self._encoder_block(64, (3, 3), (2, 2), 'relu', 'same', up4, pool_layer=False)

        # Upsample to restore the original image dimensions
        upsampled_output = keras.layers.UpSampling2D(size=(2, 2))(final_conv)
        outputs = keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(upsampled_output)

        self.model = keras.models.Model(inputs=[inputs], outputs=[outputs])

    def _encoder_block(self, num_filters, kernel_size, pool_size, activation, padding, input_layer, pool_layer=True):
        conv = keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size, activation=activation,
                                   padding=padding)(input_layer)
        conv = keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size, activation=activation,
                                   padding=padding)(conv)
        if pool_layer:
            pool = keras.layers.MaxPooling2D(pool_size)(conv)
            return conv, pool
        else:
            return conv

    def _decoder_block(self, num_filters, up_num_filters, kernel_size, up_kernel_size, up_stride, activation, padding,
                       input_layer, concat_layer):
        conv = keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size, activation=activation,
                                   padding=padding)(input_layer)
        conv = keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size, activation=activation,
                                   padding=padding)(conv)
        upconv = keras.layers.Conv2DTranspose(filters=up_num_filters, kernel_size=up_kernel_size, strides=up_stride,
                                              padding=padding)(conv)
        upconv = keras.layers.concatenate([upconv, concat_layer], axis=3)

        return conv, upconv
