import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.python.client import device_lib

# Dense Layer class, for in each Dense Block.
class DenseLayer(tf.keras.layers.Layer):

    def __init__(self, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(DenseLayer, self).__init__()
        
        self.norm1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.conv1 = layers.Conv2D(bn_size * growth_rate, kernel_size=1, strides=1, padding='same', use_bias=False)
        
        self.norm2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
        self.conv2 = layers.Conv2D(growth_rate, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def call(self, inputs):
        # "Bottleneck function"
        concatenated_features = tf.concat(inputs, axis=-1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concatenated_features)))
        bottleneck_output = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        
        if self.drop_rate > 0:
            bottleneck_output = layers.Dropout(rate=self.drop_rate)(bottleneck_output)
            
        return bottleneck_output

# Transition Layer class, for after each DenseBlock
class Transition(tf.keras.layers.Layer):
    def __init__(self, num_input_features, num_output_features):
        super(Transition, self).__init__()

        self.norm = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv = layers.Conv2D(num_output_features, kernel_size=1, strides=1, padding='same', use_bias=False)
        self.pool = layers.AveragePooling2D(pool_size=2, strides=2)
        
    def call(self, inputs):
        x = self.conv(self.relu(self.norm(inputs)))
        return self.pool(x)

# DenseBlock class
class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_layers, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(DenseBlock, self).__init__()
        
        self.layers = [DenseLayer(growth_rate, bn_size, drop_rate, memory_efficient) for _ in range(num_layers)]

    def call(self, inputs):
        features = [inputs]
        for layer in self.layers:
            new_features = layer(features)
            features.append(new_features)

        return tf.concat(features, axis=-1)

# Complete DenseNet architecture, 
# adding first layers, then all DenseBlocks + Transition Layers, and then the final layers
class DenseNet(tf.keras.Model):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                    num_init_features=64, bn_size=4, drop_rate=0.2, num_classes=20, memory_efficient=False):
        super(DenseNet, self).__init__()

        # Add first two layers of DenseNet to features
        self.features = tf.keras.Sequential([
            layers.Conv2D(num_init_features, kernel_size=7, strides=2, padding='valid', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=3, strides=2, padding='valid')
        ])

        # Add multiple denseblocks to features based on config for densenet-121 config: [6, 12, 24, 16]
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, bn_size, growth_rate, drop_rate, memory_efficient)
            self.features.add(block)
            num_features += num_layers * growth_rate
            
            # Add transition layer between denseblocks to downsample to features
            if i != len(block_config) - 1:
                trans = Transition(num_features, num_features // 2)
                self.features.add(trans)
                num_features = num_features // 2

        # Add final stages of DenseNet to features
        self.features.add(layers.BatchNormalization())
        self.features.add(layers.ReLU())

        self.avg_pool = layers.GlobalAveragePooling2D()
        self.classifier = layers.Conv2D(num_classes, kernel_size=1, activation='softmax')

    # Run the forward propagation of the model
    def call(self, inputs):
        x = self.features(inputs)
        x = self.avg_pool(x)
        return self.classifier(x)

# Returns DenseNet 121 model
def densenet(num_classes=20):
    return DenseNet(block_config=[6, 12, 24, 16], num_classes=num_classes)