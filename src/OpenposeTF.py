
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import Algorithmia
import numpy as np
import tensorflow as tf
import cv2
import time
import logging
from tensorflow.python.client import timeline

from src.common import estimate_pose, CocoPairsRender, read_imgfile, CocoColors, draw_humans
from src.networks import get_network
#from src.pose_dataset import CocoPoseLMDB

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True

client = Algorithmia.client()

def interference():
    """
    parser = argparse.ArgumentParser(description='Tensorflow Openpose Inference')
    parser.add_argument('--imgpath', type=str, default='./images/p2.jpg')
    parser.add_argument('--input-width', type=int, default=368)
    parser.add_argument('--input-height', type=int, default=368)
    parser.add_argument('--stage-level', type=int, default=6)
    parser.add_argument('--model', type=str, default='mobilenet',
                        help='cmu / mobilenet / mobilenet_accurate / mobilenet_fast')
    args = parser.parse_args()
    """
    input_height = 368
    input_width = 368
    imgpath = './images/p2.jpg'
    stage_level = 6
    model = 'mobilenet'

    input_node = tf.placeholder(tf.float32, shape=(1, input_height, input_width, 3), name='image')

    with tf.Session(config=config) as sess:
        net, _, last_layer = get_network(model, input_node, sess)

        logging.debug('read image+')
        image = read_imgfile(imgpath, input_width, input_height)
        vec = sess.run(net.get_output(name='concat_stage7'), feed_dict={'image:0': [image]})

        a = time.time()
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        pafMat, heatMat = sess.run(
            [
                net.get_output(name=last_layer.format(stage=stage_level, aux=1)),
                net.get_output(name=last_layer.format(stage=stage_level, aux=2))
            ], feed_dict={'image:0': [image]}, options=run_options, run_metadata=run_metadata
        )
        logging.info('inference- elapsed_time={}'.format(time.time() - a))

        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)
        heatMat, pafMat = heatMat[0], pafMat[0]

        logging.debug('inference+')

        avg = 0
        for _ in range(10):
            a = time.time()
            sess.run(
                [
                    net.get_output(name=last_layer.format(stage=stage_level, aux=1)),
                    net.get_output(name=last_layer.format(stage=stage_level, aux=2))
                ], feed_dict={'image:0': [image]}
            )
            logging.info('inference- elapsed_time={}'.format(time.time() - a))
            avg += time.time() - a
        logging.info('prediction avg= %f' % (avg / 10))

        '''
        logging.info('pickle data')
        with open('person3.pickle', 'wb') as pickle_file:
            pickle.dump(image, pickle_file, pickle.HIGHEST_PROTOCOL)
        with open('heatmat.pickle', 'wb') as pickle_file:
            pickle.dump(heatMat, pickle_file, pickle.HIGHEST_PROTOCOL)
        with open('pafmat.pickle', 'wb') as pickle_file:
            pickle.dump(pafMat, pickle_file, pickle.HIGHEST_PROTOCOL)
        '''

        logging.info('pose+')
        a = time.time()
        humans = estimate_pose(heatMat, pafMat)
        logging.info('pose- elapsed_time={}'.format(time.time() - a))

        logging.info('image={} heatMap={} pafMat={}'.format(image.shape, heatMat.shape, pafMat.shape))

        """
        process_img = CocoPoseLMDB.display_image(image, heatMat, pafMat, as_numpy=True)

        # display
        image = cv2.imread(imgpath)
        image_h, image_w = image.shape[:2]
        image = draw_humans(image, humans)

        scale = 480.0 / image_h
        newh, neww = 480, int(scale * image_w + 0.5)

        image = cv2.resize(image, (neww, newh), interpolation=cv2.INTER_AREA)

        convas = np.zeros([480, 640 + neww, 3], dtype=np.uint8)
        convas[:, :640] = process_img
        convas[:, 640:] = image

        cv2.imshow('result', convas)
        cv2.waitKey(0)

        tf.train.write_graph(sess.graph_def, '.', 'graph-tmp.pb', as_text=True)
        """


def load_data():
    """Retrieve variable checkpoints and graph from user collection"""
    #vc_uri = 'data://user_name/data_collection/variable_checkpoint_tensorflow.ckpt'
    #checkpoint_file = client.file(vc_uri).getFile().name

    graph_uri = 'data://bilgeckers/OpenposeTF/optimized_openpose.pb'
    graph_file = client.file(graph_uri).getFile().name

    #return (checkpoint_file, graph_file)
    return graph_file

def create_graph(graph_path):
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(graph_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

interference()


def apply(input):

    print("lalaaa")
    
    return("koekoek")
