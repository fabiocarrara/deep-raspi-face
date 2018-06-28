from __future__ import print_function
import sys
import os
import cv2
import time
import numpy as np
import skimage.transform
from sklearn.preprocessing import normalize
from webcam_video_stream import WebcamVideoStream


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


def train(args):

    if args.people and os.path.exists(args.people[0]) and os.path.exists(args.people[1]):
        pass
    else:
        raise Exception("File not found")

    # initialize camera
    if args.usePiCamera:
        from imutils.video import VideoStream
        vs = VideoStream(usePiCamera=True, framerate=args.framerate, resolution=args.resolution)
    else:
        vs = WebcamVideoStream()

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
    #MODEL = os.path.join(SCRIPT_DIR, 'extractor/caffe_models/{0}_caffe/{0}'.format(args.model))
    MODEL = os.path.join(SCRIPT_DIR, '/home/fabiovalerio/faces/caffe_model/{0}_caffe/{0}'.format(args.model))
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

    features = []
    n = 0
    while n < 10:
        start_whole = time.time()
        # capture image from camera
        start = time.time()
        img = vs.read()
        if img is None: continue  # skip initial empty frames due to camera init. delay
        end = time.time()
        capture_time = end - start
        print('\n\tCapture:', capture_time, 's')

        blur = cv2.GaussianBlur(img,(5,5),0)
        img = cv2.addWeighted(blur,1.5,img,-0.5,0)

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

        bb = []
        matches = []

        if len(faces) == 0:
            print("No faces detected")
        if len(faces) > 1:
            print("Too many faces on the image")

        if len(faces) == 1:
            face = faces[0]
            if args.usePiCamera:
                face *= np.tile(args.resolution, 2)
            else:
                face *= np.tile(img.shape[:2][::-1], 2)

            (startX, startY, endX, endY) = face.astype("int")
            bb.append((startX, startY, endX, endY))

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

            descriptor = normalize(descriptor.reshape(1,-1))
            features.append(descriptor)
            n += 1
            print("************************************* acquired faces: {}".format(n))
            if len(faces) != 0:
                for i in range(len(bb)):
            	    cv2.rectangle(img, bb[i][:2], bb[i][2:4], (0, 255, 0), 2)
                for match in matches:
                    cv2.putText(img,
                                match,
                               (50, 50*(i+1)), # x, y of top left corner of text
                               cv2.FONT_HERSHEY_SIMPLEX,
                               0.7,
                               (0, 255, 0), # BGR
                               2)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.imshow('frame',gray)
                cv2.waitKey(1)

            time.sleep(1)

    person_id = raw_input("Insert person id: ")

    with open(args.people[0], 'a') as f:
        f.write(person_id)
        f.write("\n")

    features = np.array(features).squeeze()
    print(features.shape)
    with open(args.people[1], 'a') as f:
        np.savetxt(f, features, delimiter=' ')

