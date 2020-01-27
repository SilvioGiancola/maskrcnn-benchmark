from flask import Flask, request, jsonify

import os
import sys
import numpy as np
import json
import cv2

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.structures.bounding_box import BoxList
from predictor import COCODemo

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

DATA_FOLDER = "/media/giancos/Football/CloudLabeling/"

class FlaskServer(object):
    def __init__(self, args):
        self.model = None
        self.PORT = args.PORT
        self.HOST = args.HOST

    def setup(self, args):
        self.project = "Seeds_Striga_Strategy2"  # project
        self.architecture = "R_50_C4_1x"  # architecture for FasterRCNN
        self.model = None # FasterRCNN model

        print(list_project())
        print(list_architecture())

        initialize_model()
        
        app.run(host=self.HOST, port=self.PORT)


app = Flask(__name__)


@app.route("/")  # main webpage
def home():
    return "Hello world!"

@app.route("/api/list_projects", methods=['POST'])
def list_project():
    return os.listdir(DATA_FOLDER)  # ["Seeds", "Pills", "Spine", "Fish"]


@app.route("/api/list_architecture", methods=['POST'])
def list_architecture():
    folder_architectures = os.path.join(DATA_FOLDER, server.project, "output")
    return os.listdir(folder_architectures)


@app.route("/api/select_project", methods=['POST'])
def select_project():
    headers = request.headers
    print(headers)
    server.project = headers
    server.architecture = list_architecture()[0]
    err = initialize_model()

    result = jsonify({'Error': err,
                      })
    print(result)
    return result


@app.route("/api/select_architecture", methods=['POST'])
def select_architecture():
    headers = request.headers
    print(headers)
    server.architecture = headers
    err = initialize_model()

    result = jsonify({'Error': err,
                      })
    print(result)
    return result


def initialize_model():
    # Initialize the model
    # args: None
    # return: [boolean]: Error loading model

    config_file = f"./configs/e2e_faster_rcnn_{server.architecture}.yaml"
    print(f'trying to load model from {config_file}')
    if not os.path.exists(config_file):
        print("Dir does not exists")
        return True

    cfg.merge_from_file(config_file)

    cfg.DATASETS.DATA_DIR = os.path.join(DATA_FOLDER, server.project)
    cfg.DATASETS.TRAIN = ['folder_train']
    cfg.DATASETS.TEST = ['folder_test']
    cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES=3
    cfg.TEST.DETECTIONS_PER_IMG=200
    cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG=200
    cfg.OUTPUT_DIR = os.path.join(
        cfg.DATASETS.DATA_DIR, "output", server.architecture)
        # cfg.DATASETS.DATA_DIR, "output", server.architecture)

    # cfg.freeze()

    print(cfg.OUTPUT_DIR + "/model_bestval.pth")
    server.model = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.5,
        weight_loading=cfg.OUTPUT_DIR + "/model_bestval.pth",
    )

    # Get list of CATEGORIES from dataloader
    data_loader = make_data_loader(cfg, is_train=False)
    server.model.CATEGORIES = data_loader[0].dataset.CLASSES

    return False


@app.route("/api/predict", methods=['POST'])
def predict():

    # if server.model is None:
    initialize_model()

    print("List Projects:", list_project())
    print("List Architecture:", list_architecture())

    headers = request.headers
    print(headers)
    print("got something")
    if (headers["content-type"] == "image/jpeg") and server.model is not None:
        # Read request
        image_bytes = request.data
        nparr = np.frombuffer(image_bytes,np.uint8)
        image_bytes = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # compute prediction
        predictions = server.model.compute_prediction(image_bytes)
        predictions = server.model.select_top_predictions(predictions)
        print(predictions)
                   
        boxes = predictions.bbox.numpy().tolist()
        labels_words = [server.model.CATEGORIES[label]
                        for label in predictions.get_field("labels").numpy().tolist()]
        scores = predictions.get_field("scores").numpy().tolist()
        
        return jsonify({'boxes': boxes,
                        'labels_words': labels_words,
                        'scores': scores
                        })


if __name__ == "__main__":

    parser = ArgumentParser(
        description='Train or test Shape Completion for 3D Tracking',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--HOST", type=str, default="localhost",
                        help="host IP")
    parser.add_argument("--PORT", type=int, default=5000,
                        help="commmunication port")

    parser.add_argument(
        '--GPU',
        required=False,
        type=int,
        default=-1,
        help='ID of the GPU to use')

    args = parser.parse_args()

    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    server = FlaskServer(args)
    server.setup(sys.argv[1:])
