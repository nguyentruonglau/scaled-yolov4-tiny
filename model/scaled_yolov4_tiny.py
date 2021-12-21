from generator.config_data import yolo_anchors
import tensorflow as tf


def box_decode(encoded_pred_boxes, args, grid_index):
    '''Decoded boxes, shape is [batch,N,4]
    '''
    stride = 16 * 2 ** grid_index
    (batch_size, grid_height, grid_width) = tf.shape(encoded_pred_boxes)[0:3]
    normalized_anchors = tf.cast(yolo_anchors[grid_index],tf.dtypes.float32)/tf.cast(tf.shape(encoded_pred_boxes)[1:3]*stride,tf.dtypes.float32)

    grid_xy = tf.stack(tf.meshgrid(tf.range(grid_width), tf.range(grid_height)), axis=-1)
    grid_xy = tf.cast(tf.expand_dims(grid_xy, axis=-2),tf.dtypes.float32)

    scales_x_y = tf.cast(args.scales_x_y[grid_index], tf.dtypes.float32)
    decoded_pred_cxcy = (grid_xy + encoded_pred_boxes[..., 0:2]*scales_x_y - 0.5 * (scales_x_y - 1))/(grid_width,grid_height)

    decoded_pred_wh = (encoded_pred_boxes[..., 2:4]*2)**2 *normalized_anchors

    half_decoded_pred_wh = decoded_pred_wh/2
    decoded_x1y1x2y2 = tf.clip_by_value(tf.concat([decoded_pred_cxcy - half_decoded_pred_wh, decoded_pred_cxcy+half_decoded_pred_wh],axis=-1),0,1.)
    return tf.reshape(decoded_x1y1x2y2, [batch_size, -1, 4])


def postprocess(outputs, args):
    """Decode the output of model, return box and score respectively
    """
    num_classes = int(args.num_classes)
    if num_classes == 1: num_classes = 0
    boxes_list = []; scores_list = []

    for index, output in enumerate(outputs):
        output = tf.reshape(output, [tf.shape(output)[0], tf.shape(output)[1], tf.shape(output)[2], -1, 5+num_classes])
        output = tf.sigmoid(output)
        decoded_boxes = box_decode(output[..., 0:4], args, index)

        if num_classes == 0: scores = output[..., 4:5]
        else: scores = output[..., 4:5] * output[..., 5:]

        scores = tf.reshape(scores, [tf.shape(scores)[0], -1, tf.shape(scores)[-1]])
        boxes_list.append(decoded_boxes); scores_list.append(scores)

    decoded_boxes = tf.concat(boxes_list, axis=-2,name='output_boxes')
    scores = tf.concat(scores_list, axis=-2,name='output_scores')

    return decoded_boxes, scores


def conv2d_bn_leaky(x, filters, kernel_size, strides=(1,1), padding='same',name=None):
    """Convolution + batch normalization + Leaky activation
    """
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding, use_bias=False, name=name+"_conv2d")(x)
    x = tf.keras.layers.BatchNormalization(name=name+"_batch_normalization")(x)
    return tf.keras.layers.LeakyReLU(alpha=0.1)(x)


def tiny_block(x,name):
    """Tiny block for backbone of scaled yolov4 tiny
    """
    x = conv2d_bn_leaky(x, x.shape[-1],(3, 3),name=name+"_1")
    x1 = x[..., x.shape[-1]//2:]
    x2 = conv2d_bn_leaky(x1, x1.shape[-1], (3, 3),name=name+"_2")
    x3 = conv2d_bn_leaky(x2, x2.shape[-1], (3, 3),name=name+"_3")
    x3 = tf.keras.layers.Concatenate()([x3,x2])
    x3 = conv2d_bn_leaky(x3, x3.shape[-1], (1, 1),name=name+"_4")
    x4 = tf.keras.layers.Concatenate()([x, x3])
    return x4,x3


def backbone(x):
    """Backbone of scaled yolov4 tiny, contain many tiny_block
    """
    x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = conv2d_bn_leaky(x,32,(3,3),strides=(2,2),padding='valid',name="block_1_1")
    x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = conv2d_bn_leaky(x,64,(3,3),strides=(2,2),padding='valid',name="block_2_1")

    x,_ = tiny_block(x,name="block_3")
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x,_ = tiny_block(x,name="block_4")
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x,x1 = tiny_block(x,name="block_5")
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = conv2d_bn_leaky(x, x.shape[-1], (3, 3), strides=(1, 1), padding='same',name="block_5_5")
    output1 = conv2d_bn_leaky(x, x.shape[-1]//2, (1, 1), strides=(1, 1), padding='same',name="block_5_6")
    x = conv2d_bn_leaky(output1, output1.shape[-1] // 2, (1, 1), strides=(1, 1), padding='same',name="block_5_7")
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
    output2 = tf.keras.layers.Concatenate()([x, x1])
    return [output2, output1]


def head(inputs, args):
    """Head of scaled yolov4 tiny, contain 2 scale for detection
    """
    class_num = int(args.num_classes)
    if class_num == 1: class_num = 0
    output_layers = []
    head_conv_filters = [256, 512]

    for index, x in enumerate(inputs):
        x = conv2d_bn_leaky(x, head_conv_filters[index], (3, 3), name='yolov4_head_%d_1' % (index+1))
        x = tf.keras.layers.Conv2D(len(yolo_anchors[index]) * (class_num + 5), (1, 1), use_bias=True, name='yolov4_head_%d_2_conv2d' % (index+1))(x)
        x = tf.reshape(x, [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], -1, class_num + 5])
        output_layers.append(x)
    return output_layers


def scaled_yolov4_tiny(args, training=True):
    """Fully build model scaled yolov4 tiny
    """
    input = tf.keras.layers.Input((args.model_shape, args.model_shape, 3))
    outputs = backbone(input)
    outputs = head(outputs,args)

    if training:
        model = tf.keras.Model(inputs=input, outputs=outputs)
        return model
    
    pre_nms_decoded_boxes, pre_nms_scores = postprocess(outputs,args)
    return tf.keras.Model(inputs=input, outputs=[pre_nms_decoded_boxes, pre_nms_scores])