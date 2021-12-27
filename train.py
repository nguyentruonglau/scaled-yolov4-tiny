from model.scaled_yolov4_tiny import scaled_yolov4_tiny
from model.yolov4_losses import yolov4_loss
from model.lr_scheduler import get_lr_scheduler
from model.optimizers import yolov4_optimizers

from utility.eager_coco_map import EagerCocoMap
from utility.fit_coco_map import CocoMapCallback
from utils import load_pretrained_model
from utils import check_input_shape
from generator.data_generator import data_generator

import time, argparse, sys, os
import webbrowser, logging, warnings, platform

from tqdm import tqdm
import numpy as np
from tensorboard import program
import tensorflow as tf


# compiled using XLA, auto-clustering on GPU & CPU
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'

# re-writes the environment variables and makes only certain NVIDIA GPU(s) visible for that process.
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" #one GPU used, have id=0

# dynamic allocate GPU memory
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices: tf.config.experimental.set_memory_growth(physical_devices[0], True)
else: warnings.warn('[WARNING]: GPU not found, CPU current is being used')


def parse_args(args):
    parser = argparse.ArgumentParser()
    # training
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--start-eval-epoch', default=1, type=int, help='start running the evaluation program from epoch')
    parser.add_argument('--eval-epoch-interval', default=1, help='with default=1, evaluate each epoch')
    # model
    parser.add_argument('--model-shape', default=416, help="Input shape of model, must: %32 == 0 and >=320, understood as: model_shape=(448, 448)")
    parser.add_argument('--use-pretrain', default=True, type=bool)
    parser.add_argument('--pretrained-weights', default='./database/models/pretrained/coco_pretrain', help='path to pretrained model or checkpoint model')
    parser.add_argument('--checkpoints-dir', default='./database/models/checkpoint', help="directory to store checkpoints of model during training.")
    parser.add_argument('--best-model-path', default='./database/models/best', help='path to directory will save best model')
    # loss
    parser.add_argument('--box-regression-loss', default='ciou',help="choices=['giou','diou','ciou']")
    parser.add_argument('--classification-loss', default='bce', help="choices=['sigmoid_focal', 'bce']")
    # dataset
    parser.add_argument('--num-classes', default=1, help='number of class of dataset')
    parser.add_argument('--class-names', default='./database/dataset/Car/ClassNames/car.names', help="contain class names of dataset")
    parser.add_argument('--dataset', default='./database/dataset/Car', help='path to directory contain dataset')
    parser.add_argument('--skip-difficult', default=True)
    parser.add_argument('--train-set', default='./database/dataset/Car/ImageSets/train.txt')
    parser.add_argument('--val-set', default='./database/dataset/Car/ImageSets/val.txt')
    # optimizer
    parser.add_argument('--optimizer', default='sam_adam', help="choices=['sam_adam']")
    parser.add_argument('--weight-decay', default=5e-4)
    # lr scheduler
    parser.add_argument('--lr-scheduler', default='cosine', type=str, help="choices=['step','cosine']")
    parser.add_argument('--init-lr', default=1e-3, type=float)
    parser.add_argument('--lr-decay', default=0.1, type=float)
    parser.add_argument('--lr-decay-epoch', default=[160, 180], help='step, [169, 180] is two step of step decay')
    parser.add_argument('--warmup-epochs', default=10, type=int)
    parser.add_argument('--warmup-lr', default=1e-6, type=float)
    # postprocess
    parser.add_argument('--max-box-num-per-image', default=100)
    parser.add_argument('--nms-max-box-num', default=100)
    parser.add_argument('--nms-iou-threshold', default=0.2, type=float)
    parser.add_argument('--nms-score-threshold', default=0.05, type=float, help='confidence of object')
    # anchor match
    parser.add_argument('--anchor-match-type', default='iou',help="choices=['iou']")
    parser.add_argument('--anchor-match-iou_thr', default=0.2, type=float)
    # pyramid neural network
    parser.add_argument('--scales-x-y', default=[2., 2., 2., 2., 2.], help='reduce the resolution between laysers of the pyramid neural network')
    #tensorboard
    parser.add_argument('--tensorboard', default=True, type=bool, help='choice google webrowser to default, if you remote to server, choice False')
    # label smoothing
    parser.add_argument('--label-smooth', default=0.2, type=float, help='float in [0, 1], using 1 - 0.5 * label_smoothing \
                        for the target class and 0.5 * label_smoothing for the non-target class')
    return parser.parse_args(args)


@tf.function(experimental_relax_shapes=True)
def training(batch_imgs, batch_labels):
    """Training phase, executed with graph, (optimize with Sam Adam)
    """
    # first step
    with tf.GradientTape() as tape:
        model_outputs = model(batch_imgs, training=True)
        data_loss = 0
        for output_index, output_val in enumerate(model_outputs):
            loss = loss_fun[output_index](batch_labels[output_index], output_val)
            data_loss += tf.reduce_sum(loss)
        total_loss = data_loss + args.weight_decay * tf.add_n(
            [tf.nn.l2_loss(v) for v in model.trainable_variables
            if 'batch_normalization' not in v.name
            ]
        )
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.first_step(grads, model.trainable_variables)

    # second step
    with tf.GradientTape() as tape:
        model_outputs = model(batch_imgs, training=True)
        data_loss = 0
        for output_index, output_val in enumerate(model_outputs):
            loss = loss_fun[output_index](batch_labels[output_index], output_val)
            data_loss += tf.reduce_sum(loss)
        total_loss = data_loss + args.weight_decay * tf.add_n(
                [tf.nn.l2_loss(v) for v in model.trainable_variables 
                if 'batch_normalization' not in v.name
            ]
        )
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.second_step(grads, model.trainable_variables)

    # return total loss
    return total_loss


if __name__ == "__main__":
    args = parse_args(sys.argv[1:]);
    # check argument
    check_input_shape(args)
    # load data
    train_generator, valid_dataset = data_generator(args)
    #load model
    model = load_pretrained_model(args)
    # model.summary()

    # callback
    loss_fun = [yolov4_loss(args, grid_index) for grid_index in range(2)]
    lr_scheduler = get_lr_scheduler(args)
    optimizer = yolov4_optimizers(args)

    # remove logs file
    if platform.system() == 'Windows': os.system('rmdir /s /q logtrain')
    else: os.system('rm -rf logtrain')

    # tensorboard
    open_tensorboard_url = False
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', 'logtrain', '--reload_interval', '15'])
    url = tb.launch()
    print("[INFOR]: Tensorboard engine is running at {}".format(url))
    # dataset
    print("[INFOR]: Loading dataset...")

    start_time = time.perf_counter()
    coco_map = EagerCocoMap(valid_dataset, model, args)
    max_coco_map = -1
    max_coco_map_epoch = -1
    model_name = None
    best_weight_path = None

    # log tensorboard
    train_writer = tf.summary.create_file_writer("logtrain/Train")
    mAP_writer = tf.summary.create_file_writer("logtrain/mAP")
    lr_writer = tf.summary.create_file_writer("logtrain/Learning_Rate")

    # training
    for epoch in range(int(args.epochs)):
        lr = lr_scheduler(epoch)
        optimizer.learning_rate.assign(lr)
        remaining_epoches = args.epochs - epoch - 1
        epoch_start_time = time.perf_counter()
        train_loss = 0
        train_generator_tqdm = tqdm(enumerate(train_generator), total=len(train_generator))

        # loop through batch_imgs    
        for batch_index, (batch_imgs, batch_labels) in train_generator_tqdm:
            train_loss += training(batch_imgs, batch_labels)
            train_generator_tqdm.set_description(
                "Epoch: {}/{}, train_loss: {:.4f}, lr: {:.6f}".format(
                    epoch, args.epochs,
                    train_loss / (batch_index + 1),
                    optimizer.learning_rate.numpy()
                )
            )
        train_generator.on_epoch_end()

        with train_writer.as_default():
            #flush train loss
            tf.summary.scalar("train_loss", train_loss/len(train_generator), step=epoch)
            train_writer.flush()
            #flush learning rate
            tf.summary.scalar("learning_rate", optimizer.learning_rate.numpy(), step=epoch)
            lr_writer.flush()

        # evaluation
        if epoch >= args.start_eval_epoch:
            if epoch % args.eval_epoch_interval == 0:
                summary_metrics = coco_map.eval()
                if summary_metrics['Precision/mAP@.50IOU'] > max_coco_map:
                    max_coco_map = summary_metrics['Precision/mAP@.50IOU']
                    max_coco_map_epoch = epoch
                    model_name = 'best_weight_{}_{}_{:.3f}'.format(args.model_type, max_coco_map_epoch, max_coco_map)
                    best_weight_path = os.path.join(args.checkpoints_dir, model_name)
                    model.save_weights(best_weight_path)

                print("Max_CoCo_mAP:{} achieved at Epoch: {}".format(max_coco_map, max_coco_map_epoch))
                with mAP_writer.as_default():
                    tf.summary.scalar("mAP@0.5", summary_metrics['Precision/mAP@.50IOU'], step=epoch)
                    mAP_writer.flush()

        cur_time = time.perf_counter()
        one_epoch_time = cur_time - epoch_start_time
        print("Time elapsed: {:.3f} hour, time left: {:.3f} hour\n\n".format((cur_time-start_time)/3600, 
            remaining_epoches*one_epoch_time/3600))
        if epoch>0 and not open_tensorboard_url:
            open_tensorboard_url = True
            webbrowser.open(url,new=1)
    print("[INFOR]: Training is finished!")

    print("[INFOR]: Exporting model...")
    if args.best_model_path and best_weight_path:
        tf.keras.backend.clear_session()
        model = scaled_yolov4_tiny(args, training=False)
        model.load_weights(best_weight_path)
        tf.saved_model.save(model, os.path.join(args.best_model_path, model_name.replace('weight', 'model')))
    print("[INFOR]: Export model successfully, find your model at {}!".format(args.best_model_path))