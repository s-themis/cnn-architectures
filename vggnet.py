from tensorflow.keras import layers, Model


def vgg16():
    def conv_block(x, conv_layers, filters):
        for _ in range(conv_layers):
            x = layers.Conv2D(filters=filters,
                              kernel_size=3,
                              padding="same",
                              activation="relu")(x)
        x = layers.MaxPool2D(pool_size=2, strides=2)(x)
        return x

    # Input.
    x_in = layers.Input(shape=(224, 224, 3))

    # 1st convolutional block.
    x = conv_block(x_in, conv_layers=2, filters=64)

    # 2nd convolutional block.
    x = conv_block(x, conv_layers=2, filters=128)

    # 3rd convolutional block.
    x = conv_block(x, conv_layers=3, filters=256)

    # 4th convolutional block.
    x = conv_block(x, conv_layers=3, filters=512)

    # 5th convolutional block.
    x = conv_block(x, conv_layers=3, filters=512)

    # Fully-connected layers.
    x = layers.Flatten()(x)
    x = layers.Dense(units=4096, activation="relu")(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(units=4096, activation="relu")(x)
    x = layers.Dropout(rate=0.5)(x)

    # Output.
    x_out = layers.Dense(units=1000, activation="softmax")(x)

    # Model.
    model = Model(inputs=x_in, outputs=x_out)

    return model


if __name__ == "__main__":

    from tensorflow.keras.utils import plot_model

    model = vgg16()
    model.summary()
    plot_model(model, to_file="vgg16.png", show_shapes=True)
