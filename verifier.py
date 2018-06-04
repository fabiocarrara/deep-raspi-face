from __future__ import print_function
import sys
import os
os.environ['GLOG_minloglevel'] = '2'

import picamera
import picamera.array
import cv2
import time
import numpy as np
import skimage.transform
import caffe

print("Required modules imported.")

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


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


if __name__ == '__main__':

    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    MODEL = os.path.join(SCRIPT_DIR, 'caffe_models/senet50_ft_caffe/senet50_ft')
    NET_DEF = '{}.prototxt'.format(MODEL)
    NET_WEIGHTS = '{}.caffemodel'.format(MODEL)
    LAYER = 'pool5/7x7_s1'  # 'classifier'

    mean = np.array([91.4953, 103.8827, 131.0912]).reshape(3, 1, 1)

    # initialize the neural net
    start = time.time()
    net = caffe.Net(NET_DEF, caffe.TEST, weights=NET_WEIGHTS)
    end = time.time()
    eprint('Net loading time:', (end - start))

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    with picamera.PiCamera(framerate=1, resolution=(1280, 960)) as camera:
        camera.start_preview()
        while True:
            with picamera.array.PiRGBArray(camera) as stream:
                # capture image from camera
                start = time.time()
                camera.capture(stream, format='bgr')
                # At this point the image is available as stream.array
                img = stream.array
                end = time.time()
                eprint('Image capturing time:', (end - start), img.shape)
            
            # detect faces
            start = time.time()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.25, 6) #, minSize=75, maxSize=500)
            end = time.time()
            eprint('Face Detection time:', (end - start), len(faces))
            del gray
            
            for (x,y,w,h) in faces:
                face = img[y:y+h, x:x+w]
                
                # transform image       
                start = time.time()
                face = preprocess_image(face, mean)
                end = time.time()
                eprint('Image Pre-processing time:', (end - start), face.shape)

                # run the net and return prediction
                net.blobs['data'].data[...] = face
                start = time.time()
                results = net.forward(end=LAYER)[LAYER]
                end = time.time()
                eprint('Prediction time:', (end - start))

                # turn it into something we can play with and examine which is in a multi-dimensional array
                # print(results)
                np.savetxt(sys.stdout, results, fmt='%g')

