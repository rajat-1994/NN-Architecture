'''
# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- https://github.com/KaimingHe/deep-residual-networks
'''
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, Activation,ZeroPadding2D
from keras.layers import GlobalAveragePooling2D,GlobalMaxPooling2D,AveragePooling2D,MaxPooling2D
from keras.optimizers import RMSprop
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K

#this is the first block of every stage
def conv_block(input_tensor,kernel_size,filters,strides=(2,2)):
    
    f1,f2,f3=filters
    #Bottleneck conv 1x1
    x = Conv2D(f1,kernel_size=(1,1),strides=strides)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #3x3 conv
    x = Conv2D(f2,kernel_size=kernel_size,strides=(1, 1),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #Bottleneck 1x1 conv
    x = Conv2D(f3,kernel_size=(1,1),strides=(1, 1))(x)
    x = BatchNormalization()(x)
    
    shortcut = Conv2D(f3,kernel_size=(1,1),strides=strides,padding='same')(input_tensor)
    shortcut = BatchNormalization()(shortcut)
    
        
    x=add([x,shortcut])
    x = Activation('relu')(x)
    
    return x
    

def identity_block(input_tensor,kernel_size,filters):
    
    f1,f2,f3=filters
    #Bottleneck conv 1x1
    x = Conv2D(f1,kernel_size=(1,1),strides=(1, 1))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #3x3 conv
    x = Conv2D(f2,kernel_size=kernel_size,strides=(1, 1),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #Bottleneck 1x1 conv
    x = Conv2D(f3,kernel_size=(1,1),strides=(1, 1))(x)
    x = BatchNormalization()(x)
    
    
    x=add([x,input_tensor])
    x = Activation('relu')(x)
    return x

def resnet50(input_shape=(224,224,3),pooling='avg'):
    
    inp=Input(shape=input_shape)
    bn_axis=3
    
    filters=64
    blocks=[3,4,6,3]
    
    x= ZeroPadding2D(padding=(3,3))(inp)
    x = Conv2D(64,kernel_size=(7,7),strides=(2, 2),padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(x)

    #stage 1
    x = conv_block(x, 3, [64, 64, 256], strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256])
    x = identity_block(x, 3, [64, 64, 256])
    #stage 2
    x = conv_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
    #stage 3
    x = conv_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    #stage 4
    x = conv_block(x, 3, [512, 512, 2048])
    x = identity_block(x, 3, [512, 512, 2048])
    x = identity_block(x, 3, [512, 512, 2048])
    
    x = AveragePooling2D((7, 7))(x)
    
    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D()(x)

    model = Model(inp, x, name='resnet50')
    
    return model
    
def resnet101(input_shape=(224,224,3),pooling='avg'):
    
    inp=Input(shape=input_shape)
    bn_axis=3
    
    filters=64
    blocks=[3,4,6,3]
    
    x= ZeroPadding2D(padding=(3,3))(inp)
    x = Conv2D(64,kernel_size=(7,7),strides=(2, 2),padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(x)

    #stage 1
    x = conv_block(x, 3, [64, 64, 256], strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256])
    x = identity_block(x, 3, [64, 64, 256])
    #stage 2
    x = conv_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
    #stage 3
    x = conv_block(x, 3, [256, 256, 1024])
    for i in range(0,22):
        x = identity_block(x, 3, [256, 256, 1024])

    #stage 4
    x = conv_block(x, 3, [512, 512, 2048])
    x = identity_block(x, 3, [512, 512, 2048])
    x = identity_block(x, 3, [512, 512, 2048])
    
    x = AveragePooling2D((7, 7))(x)
    
    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D()(x)

    model = Model(inp, x, name='resnet50')
    
    return model
    
    
def resnet152(input_shape=(224,224,3),pooling='avg'):
    
    inp=Input(shape=input_shape)
    bn_axis=3
    
    filters=64
    blocks=[3,4,6,3]
    
    x= ZeroPadding2D(padding=(3,3))(inp)
    x = Conv2D(64,kernel_size=(7,7),strides=(2, 2),padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(x)

    #stage 1
    x = conv_block(x, 3, [64, 64, 256], strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256])
    x = identity_block(x, 3, [64, 64, 256])
    #stage 2
    x = conv_block(x, 3, [128, 128, 512])
    for i in range(0,7):    
        x = identity_block(x, 3, [128, 128, 512])

    #stage 3
    x = conv_block(x, 3, [256, 256, 1024])
    for i in range(0,35):
        x = identity_block(x, 3, [256, 256, 1024])

    #stage 4
    x = conv_block(x, 3, [512, 512, 2048])
    x = identity_block(x, 3, [512, 512, 2048])
    x = identity_block(x, 3, [512, 512, 2048])
    
    x = AveragePooling2D((7, 7))(x)
    
    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D()(x)

    model = Model(inp, x, name='resnet50')
    
    return model
    
    
    