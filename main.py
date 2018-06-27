import argparse
from face_training.py import train
from face_detection import recognize

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Deep Face Verifier', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-t', '--training', action='store_true', help='Acquire new faces (defualt: False)')
    parser.add_argument('-m', '--model', default='resnet50_ft', choices=['senet50_ft', 'senet50_scratch', 'resnet50_ft', 'resnet50_scratch'])
    parser.add_argument('-pc', '--usePiCamera', action='store_true', help='If True it uses piCamera. If false it uses openCV VideoCapture (default: False)')
    parser.add_argument('-r', '--resolution', nargs=2, default=(1280, 960), type=int, help='capture resolution (W H)')
    parser.add_argument('-f', '--framerate', default=1, type=int, help='capture framerate')
    parser.add_argument('--detection-confidence', '--det', type=float, default=0.5, help='minimum probability to filter weak detections')
    parser.add_argument('--match-confidence', '--match', type=float, default=0.417, help='minimum confidence to accept authentication')
    parser.add_argument('-s', '--side', default=224, type=int, help='face side for feature extraction')
    parser.add_argument('-p', '--people', nargs=2, help='authenticated people\'s ID and features')
    parser.add_argument('-k', default=10, type=int, help='k for kNN classification')
    parser.add_argument('--use-sklearn-knn', action='store_true', help='use sklearn kNN classifier')

    parser.set_defaults(usePiCamera=False)
    parser.set_defaults(use_sklearn_knn=False)

    args = parser.parse_args()

    if args.training:
        train(args)
    else:
        recognize(args)

