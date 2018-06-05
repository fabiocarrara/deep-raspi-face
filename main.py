from __future__ import print_function
import sys
import os
os.environ['GLOG_minloglevel'] = '2'

import argparse
import picamera
import picamera.array
import cv2
import time
import numpy as np
import skimage.transform
import caffe

from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier as kNN

print("Required modules imported.")

def eprint(*args, **kwargs):
    # print(*args, file=sys.stderr, **kwargs)
    pass


def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]


def rescale(img, input_height, input_width):
    eprint("Original image shape:" + str(img.shape) + " and remember it should be in H, W, C!")
    eprint("Model's input shape is %dx%d" % (input_height, input_width))
    aspect = img.shape[1]/float(img.shape[0])
    eprint("Orginal aspect ratio: " + str(aspect))
    
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

    eprint("New image shape:" + str(imgScaled.shape) + " in HWC")
    return imgScaled


def preprocess_image(img, mean, side=224):
    img = skimage.img_as_float(img).astype(np.float32)
    img = rescale(img, side, side)
    img = crop_center(img, side, side)
    eprint("After crop: " , img.shape)

    # switch to CHW
    img = img.swapaxes(1, 2).swapaxes(0, 1)
    # switch to BGR not needed, PiCamera already captures in BGR
    # img = img[(2, 1, 0), :, :]
    # remove mean for better results
    img = img * 255 - mean
    # add batch size
    img = img[np.newaxis, :, :, :].astype(np.float32)
    eprint("NCHW: ", img.shape)
    return img


def main(args):

    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    MODEL = os.path.join(SCRIPT_DIR, 'extractor/caffe_models/{0}_caffe/{0}'.format(args.model))
    EXT_NET_DEF = '{}.prototxt'.format(MODEL)
    EXT_NET_WEIGHTS = '{}.caffemodel'.format(MODEL)
    LAYER = 'pool5/7x7_s1'  # 'classifier'

    mean = np.array([91.4953, 103.8827, 131.0912]).reshape(3, 1, 1)

    if args.people:
        with open(args.people[0]) as f:
            people = [line.rstrip() for line in f]
        people = np.array(people)
        auth_descriptors = np.loadtxt(args.people[1], dtype=np.float32)
        auth_id = np.arange(len(people)).repeat(10)
        print('Authorized Features:', auth_descriptors.shape)
        knn = kNN(args.k, weights='distance', n_jobs=4)
        knn.fit(auth_descriptors, auth_id)

    # initialize face detector
    """
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    if detector.empty():
        eprint('Detector not loaded properly.')
        return
    """
    
    DET_NET_DEF = 'detector/res10_300x300_ssd.prototxt'
    DET_NET_WEIGHTS = 'detector/res10_300x300_ssd_iter_140000.caffemodel'
    
    start = time.time()
    detector = cv2.dnn.readNetFromCaffe(DET_NET_DEF, DET_NET_WEIGHTS)
    end = time.time()
    print('Detector loading time:', (end - start))
    
    # initialize the neural net
    start = time.time()
    extractor = caffe.Net(EXT_NET_DEF, caffe.TEST, weights=EXT_NET_WEIGHTS)
    # extractor = cv2.dnn.readNetFromCaffe(EXT_NET_DEF, EXT_NET_WEIGHTS)
    end = time.time()
    print('Extractor loading time:', (end - start))    

    with picamera.PiCamera(framerate=args.framerate, resolution=args.resolution) as camera:
        # camera.start_preview()
        while True:
            start_whole = time.time()
            with picamera.array.PiRGBArray(camera) as stream:
                # capture image from camera
                start = time.time()
                camera.capture(stream, format='bgr')
                # At this point the image is available as stream.array
                img = stream.array
                end = time.time()
                capture_time = end - start
                print('\n\tCapture:', capture_time, 's')
            
            # detect faces
            start = time.time()
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # faces = detector.detectMultiScale(gray, 1.25, 6) #, minSize=75, maxSize=500)
            img_det = cv2.resize(img, (300, 300))
            img_det = cv2.dnn.blobFromImage(img_det, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False)
            detector.setInput(img_det)
            faces = detector.forward().squeeze()
            end = time.time()
            del img_det
            detection_time = end - start
            
            confidences = faces[:, 2]
            faces = faces[confidences > args.confidence, 3:7]
            
            print('\tDetect :', detection_time, 's,', len(faces), 'faces')
            # camera.annotate_text = 'Faces: {}'.format(len(faces))
            
            for face in faces:
                face *= np.tile(args.resolution, 2)
                (startX, startY, endX, endY) = face.astype("int")
                face = img[startY:endY, startX:endX]
                
                # transform image       
                start = time.time()
                face = preprocess_image(face, mean)
                end = time.time()
                preproc_time = end - start
                print('\tPreProc:', preproc_time, 's')

                # run the net and return prediction
                extractor.blobs['data'].data[...] = face
                # extractor.setInput(face)
                start = time.time()
                descriptor = extractor.forward(end=LAYER)[LAYER].squeeze()
                # descriptor = extractor.forward(LAYER)
                end = time.time()
                extraction_time = end - start
                print('\tExtract:', extraction_time, 's')
                # print(descriptor.shape)
                # np.savetxt(sys.stdout, descriptor, fmt='%g')
                
                if args.people:
                    start = time.time()
                    descriptor = normalize(descriptor.reshape(1,-1))
                    person_id = knn.predict(descriptor)
                    end = time.time()
                    match_time = end - start
                    print('\tMatch  :', match_time, 's,', people[person_id])
                    
                    end_whole = time.time()
                    whole = end_whole - start_whole
                    print('\tTOTAL  :', whole, 's')
                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Face Verifier', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', default='resnet50_ft', choices=['senet50_ft', 'senet50_scratch', 'resnet50_ft', 'resnet50_scratch'], help='feature extractor')
    parser.add_argument('-r', '--resolution', nargs=2, default=(1280, 960), type=int, help='capture resolution (W H)')
    parser.add_argument('-f', '--framerate', default=1, type=int, help='capture framerate')
    parser.add_argument('-c', '--confidence', type=float, default=0.5, help='minimum probability to filter weak detections')
    parser.add_argument('-p', '--people', nargs=2, help='authenticated people\'s ID and features')
    parser.add_argument('-k', default=10, type=int, help='k for kNN classification')
    args = parser.parse_args()
    
    main(args)