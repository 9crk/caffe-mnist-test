#!/usr/bin/env python
# mail:    admin@9crk.com
# author:  9crk.from China.ShenZhen
# time:    2017-03-22

import sys
if ( len(sys.argv)!=2 or sys.argv[1]=="--help" ):
    print "Usage: python <name of this script> <image to test net> \nFor ex: python ./test_mnist.py ./test.png"
    sys.exit(0)

import caffe
import numpy as np
import cv2

model = 'examples/mnist/lenet.prototxt';
weights = 'examples/mnist/lenet_iter_10000.caffemodel';
net = caffe.Net(model,weights,caffe.TEST);
caffe.set_mode_cpu()
#img = caffe.io.load_image(sys.argv[1], color=False)
img = cv2.imread(sys.argv[1],0)
if img.shape != [28,28]:
    img2 = cv2.resize(img,(28,28))
    img = img2.reshape(28,28,-1);
else:
    img = img.reshape(28,28,-1);
#revert the image,and normalize it to 0-1 range
img = 1.0 - img/255.0
out = net.forward_all(data=np.asarray([img.transpose(2,0,1)]))

print out['prob'][0]
print "My answer is: "+str(out['prob'][0].argmax())
