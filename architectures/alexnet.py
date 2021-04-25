from tensorflow.keras import layers, Model


def alexnet():

    # Input layer.
    x_in = layers.Input(shape=(224, 224, 3))

    # 1st convolutional layer.
    x = layers.Conv2D(filters=96,
                      kernel_size=11,
                      strides=4,
                      padding="same",
                      activation="relu")(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)

    # 2nd convolutional layer.
    x = layers.Conv2D(filters=256,
                      kernel_size=5,
                      padding="same",
                      activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)

    # 3rd convolutional layer.
    x = layers.Conv2D(filters=384,
                      kernel_size=3,
                      padding="same",
                      activation="relu")(x)

    # 4th convolutional layer.
    x = layers.Conv2D(filters=384,
                      kernel_size=3,
                      padding="same",
                      activation="relu")(x)

    # 5th convolutional layer.
    x = layers.Conv2D(filters=256,
                      kernel_size=3,
                      padding="same",
                      activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=3, strides=2)(x)

    # Fully-connected layers.
    x = layers.Flatten()(x)
    x = layers.Dense(units=4096, activation="relu")(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(units=4096, activation="relu")(x)
    x = layers.Dropout(rate=0.5)(x)

    # Output layer.
    x_out = layers.Dense(units=1000, activation="softmax")(x)

    # Model.
    model = Model(inputs=x_in, outputs=x_out)

    return model


if __name__ == "__main__":

    from tensorflow.keras.utils import plot_model

    model = alexnet()
    model.summary()
    plot_model(model, to_file="alexnet.png", show_shapes=True)
