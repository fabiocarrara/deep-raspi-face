from __future__ import print_function
import sys
import os

import argparse
import cv2
import time
import numpy as np
import skimage.transform

from imutils.video import VideoStream
from sklearn.preprocessing import normalize

print("Required modules imported.")

def nprint(*args, **kwargs):
    # print(*args, file=sys.stderr, **kwargs)
    pass


def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]


def rescale(img, input_height, input_width):
    nprint("Original image shape:" + str(img.shape) + " and remember it should be in H, W, C!")
    nprint("Model's input shape is %dx%d" % (input_height, input_width))
    aspect = img.shape[1] / float(img.shape[0])
    nprint("Orginal aspect ratio: " + str(aspect))
    
    if(aspect>1):
        # landscape orientation - wide image
        res = int(aspect * input_height)
        imgScaled = skimage.transform.resize(img, (input_width, res))
    if(aspect<1):
        # portrait orientation - tall image
        res = int(input_width / aspect)
        imgScaled = skimage.transform.resize(img, (res, input_height))
    if(aspect == 1):
        imgScaled = skimage.transform.resize(img, (input_width, input_height))

    nprint("New image shape:" + str(imgScaled.shape) + " in HWC")
    return imgScaled


def preprocess_image(img, mean, side=224):
    img = skimage.img_as_float(img).astype(np.float32)
    img = rescale(img, side, side)
    img = crop_center(img, side, side)
    nprint("After crop: " , img.shape)

    # switch to CHW
    img = img.swapaxes(1, 2).swapaxes(0, 1)
    # switch to BGR not needed, PiCamera already captures in BGR
    # img = img[(2, 1, 0), :, :]
    # remove mean for better results
    img = img * 255 - mean
    # add batch size
    img = img[np.newaxis, :, :, :].astype(np.float32)
    nprint("NCHW: ", img.shape)
    return img


def knn_score(z, X, y, k):
    # get set of classes
    classes = np.sort(np.unique(y))
    # compute all distances between authorized set and current descriptor
    all_dists = ((X-z)**2).sum(axis=1)
    # find kNNs
    knns = np.argsort(all_dists)[:k]
    
    # keep kNN distances and labels
    knn_dists = all_dists[knns]
    knn_labels = y[knns]
    
    # compute (unnormalized) dW-kNN score for each class
    class_scores = [np.sum(knn_dists * (knn_labels == c)) for c in classes]
    # get class with higher score
    predicted_class = classes[np.argmax(class_scores)]
    # get 1 - smallest distance in the predicted class as confidence
    confidence = 1 - knn_dists[knn_labels == predicted_class][0]

    return predicted_class, confidence


def main(args):

    if args.people:
        with open(args.people[0]) as f:
            people = [line.rstrip() for line in f]
        people = np.array(people)
        
        auth_descriptors = np.loadtxt(args.people[1], dtype=np.float32)
        auth_id = np.arange(len(people)).repeat(10)
        print('Authorized Features:', auth_descriptors.shape)
        
        if args.use_sklearn_knn:
            from sklearn.neighbors import KNeighborsClassifier as kNN
            knn = kNN(args.k, weights='distance', n_jobs=4)
            knn.fit(auth_descriptors, auth_id)

    # initialize camera
    vs = VideoStream(usePiCamera=True, framerate=args.framerate, resolution=args.resolution)
    vs.start()

    # initialize face detector
    DET_NET_DEF = 'detector/res10_300x300_ssd.prototxt'
    DET_NET_WEIGHTS = 'detector/res10_300x300_ssd_iter_140000.caffemodel'
    
    start = time.time()
    detector = cv2.dnn.readNetFromCaffe(DET_NET_DEF, DET_NET_WEIGHTS)
    end = time.time()
    print('Detector loading time:', (end - start))
    
    # initialize the extraction network
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    MODEL = os.path.join(SCRIPT_DIR, 'extractor/caffe_models/{0}_caffe/{0}'.format(args.model))
    EXT_NET_DEF = '{}.prototxt'.format(MODEL)
    EXT_NET_WEIGHTS = '{}.caffemodel'.format(MODEL)
    LAYER = 'pool5/7x7_s1'  # 'classifier'
    mean = np.array([91.4953, 103.8827, 131.0912]).reshape(3, 1, 1)
    
    args.use_caffe = 'senet' in args.model
    
    start = time.time()
    if args.use_caffe:
	os.environ['GLOG_minloglevel'] = '2'
	import caffe
        extractor = caffe.Net(EXT_NET_DEF, caffe.TEST, weights=EXT_NET_WEIGHTS)
        extractor.blobs['data'].reshape(1, 3, args.side, args.side)
    else:
        extractor = cv2.dnn.readNetFromCaffe(EXT_NET_DEF, EXT_NET_WEIGHTS)
    
    end = time.time()
    print('Extractor loading time:', (end - start))
    
    while True:
        start_whole = time.time()
        # capture image from camera
        start = time.time()
        img = vs.read()
        if img is None: continue  # skip initial empty frames due to camera init. delay
        end = time.time()
        capture_time = end - start
        print('\n\tCapture:', capture_time, 's')
        
        # detect faces
        start = time.time()
        img_det = cv2.resize(img, (300, 300))
        img_det = cv2.dnn.blobFromImage(img_det, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False)
        detector.setInput(img_det)
        faces = detector.forward().squeeze()
        end = time.time()
        del img_det
        detection_time = end - start
        
        confidences = faces[:, 2]
        faces = faces[confidences > args.detection_confidence, 3:7]
        
        print('\tDetect :', detection_time, 's,', len(faces), 'faces')
        
        for face in faces:
            face *= np.tile(args.resolution, 2)
            (startX, startY, endX, endY) = face.astype("int")
            face = img[startY:endY, startX:endX]
            
            if face.size == 0:
                print('\tDiscarded empty bounding box')
                continue
            
            # preprocess face
            start = time.time()
            face = preprocess_image(face, mean, side=args.side)
            end = time.time()
            preproc_time = end - start
            print('\tPreProc:', preproc_time, 's')

            # get the description
            start = time.time()
            if args.use_caffe:
                extractor.blobs['data'].data[...] = face
                descriptor = extractor.forward(end=LAYER)[LAYER].squeeze()
            else:
                extractor.setInput(face)
                descriptor = extractor.forward(LAYER)
            end = time.time()
            extraction_time = end - start
            print('\tExtract:', extraction_time, 's')
            
            if args.people:
                start = time.time()
                descriptor = normalize(descriptor.reshape(1,-1))
                
                if args.use_sklearn_knn:  # sklearn knn
                    confidences = knn.predict_proba(descriptor)
                    person_id = np.argmax(confidences)
                    confidence = confidences[person_id]
                else:  # VIR knn
                    person_id, confidence = knn_score(descriptor, auth_descriptors, auth_id, args.k)
                
                end = time.time()
                match_time = end - start
                
                match = people[person_id] if confidence > args.match_confidence else 'Unauthorized'
                match = '{} (Conf = {:.2f})'.format(match, confidence)
                
                print('\tMatch  :', match_time, 's,', match)
                
                end_whole = time.time()
                whole = end_whole - start_whole
                print('\tTOTAL  :', whole, 's')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Face Verifier', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', default='resnet50_ft', choices=['senet50_ft', 'senet50_scratch', 'resnet50_ft', 'resnet50_scratch'], help='feature extractor')
    parser.add_argument('-r', '--resolution', nargs=2, default=(1280, 960), type=int, help='capture resolution (W H)')
    parser.add_argument('-f', '--framerate', default=1, type=int, help='capture framerate')
    parser.add_argument('--detection-confidence', '--det', type=float, default=0.5, help='minimum probability to filter weak detections')
    parser.add_argument('--match-confidence', '--match', type=float, default=0.417, help='minimum confidence to accept authentication')
    parser.add_argument('-s', '--side', default=224, type=int, help='face side for feature extraction')
    parser.add_argument('-p', '--people', nargs=2, help='authenticated people\'s ID and features')
    parser.add_argument('-k', default=10, type=int, help='k for kNN classification')
    parser.add_argument('--use-sklearn-knn', action='store_true', help='use sklearn kNN classifier')
    parser.set_defaults(use_sklearn_knn=False)
    
    args = parser.parse_args()
    
    main(args)
