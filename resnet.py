from tensorflow.keras import layers, Model


def projection_shortcut_res_block(x, filters, downsampling):

    # Calculate number of filters on bottleneck layer
    bottleneck_filters = filters / 4

    # Set strides for 1st 1x1 layer and shortcut layer
    if downsampling:
        strides = 2
    else:
        strides = 1

    # Keep block input to calculate shortcut connection
    x_in = x

    # Main connection

    # Bottleneck number of filters
    x = layers.Conv2D(filters=bottleneck_filters,
                      kernel_size=1,
                      strides=strides)(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Apply 3x3 convolution on bottlenecked filters
    x = layers.Conv2D(filters=bottleneck_filters,
                      kernel_size=3,
                      padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Expand number of filters
    x = layers.Conv2D(filters=filters, kernel_size=1)(x)
    x = layers.BatchNormalization()(x)

    # Shortcut connection (projection)
    shortcut = layers.Conv2D(filters=filters, kernel_size=1,
                             strides=strides)(x_in)
    shortcut = layers.BatchNormalization()(shortcut)

    # Add main and shortcut connections
    x_out = layers.Add()([x, shortcut])
    x_out = layers.ReLU()(x_out)

    return x_out


def identity_shortcut_res_block(x, filters):

    # Calculate number of filters on bottleneck layer
    bottleneck_filters = filters / 4

    # Keep block input to calculate shortcut connection
    x_in = x

    # Main connection

    # Bottleneck number of filters
    x = layers.Conv2D(filters=bottleneck_filters, kernel_size=1)(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Apply 3x3 convolution on bottlenecked filters
    x = layers.Conv2D(filters=bottleneck_filters,
                      kernel_size=3,
                      padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Expand number of filters
    x = layers.Conv2D(filters=filters, kernel_size=1)(x)
    x = layers.BatchNormalization()(x)

    # Shortcut connection (identity)
    shortcut = x_in

    # Add main and shortcut connections
    x_out = layers.Add()([x, shortcut])
    x_out = layers.ReLU()(x_out)

    return x_out


def res_block_stack(x, filters, res_blocks, downsampling):

    # 1st residual block has a projection shortcut
    x = projection_shortcut_res_block(x, filters, downsampling)

    # Following residual blocks have an identity shortcut
    for _ in range(res_blocks - 1):
        x = identity_shortcut_res_block(x, filters)

    return x


def resnet50():

    # Input.
    x_in = layers.Input(shape=(224, 224, 3))

    # Initial convolutional layer.
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2,
                      padding="same")(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="same")(x)

    # 1st stack of residual blocks.
    # No down-sampling here, it's already been done by previous MaxPool2D layer.
    x = res_block_stack(x, filters=256, res_blocks=3, downsampling=False)

    # 2nd stack of residual blocks.
    x = res_block_stack(x, filters=512, res_blocks=4, downsampling=True)

    # 3rd stack of residual blocks.
    x = res_block_stack(x, filters=1024, res_blocks=6, downsampling=True)

    # 4th stack of residual blocks.
    x = res_block_stack(x, filters=2048, res_blocks=3, downsampling=True)

    # Output layer.
    x = layers.GlobalAveragePooling2D()(x)
    x_out = layers.Dense(units=1000, activation="softmax")(x)

    # Model.
    model = Model(inputs=x_in, outputs=x_out)

    return model


if __name__ == "__main__":

    from tensorflow.keras.utils import plot_model

    model = resnet50()
    model.summary()
    plot_model(model, to_file="resnet50.png", show_shapes=True)
