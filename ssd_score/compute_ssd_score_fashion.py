import numpy as np
##Caffe from ssd branc
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()
from skimage import img_as_float
from tqdm import tqdm
from PIL import Image

class SSDScorer(object):
    def __init__(self, model_def='deploy.prototxt', model_weights='VGG_VOC0712_SSD_300x300_iter_120000.caffemodel'):
        self.net = caffe.Net(model_def, model_weights, caffe.TEST)

        # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_mean('data', np.array([104,117,123])) # mean pixel
        self.transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255]
        self.transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order

        # set net to batch size of 1
        self.img_side = 300
        self.net.blobs['data'].reshape(1,3,self.img_side,self.img_side)

    def get_score(self, image, image_class):
        image = img_as_float(image)
        transformed_image = self.transformer.preprocess('data', image)
        self.net.blobs['data'].data[...] = transformed_image

        detections = self.net.forward()['detection_out']

        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        top_indices = [i for i, label in enumerate(det_label) if label == image_class]
        score = 0 if len(top_indices) == 0 else max(det_conf[top_indices])
        return score

    def get_score_image_set(self, imgs, image_class=15):
        #image_class=15 Only persons
        scores = []
        for img in tqdm(imgs):
            scores.append(self.get_score(img, image_class))
        return np.mean(scores)


def addBounding(image, bound=40):
    # print(image.shape)
    h, w, c = image.shape
    image_bound = np.ones((h, w+bound*2, c))*255
    image_bound = image_bound.astype(np.uint8)
    image_bound[:, bound:bound+w] = image

    return image_bound

if __name__ == "__main__":
    from skimage.io import imread
    import os
    from argparse import ArgumentParser

    split = lambda s: tuple(map(int, s.split(',')))
    parser = ArgumentParser(description="Computing ssd_score")
    parser.add_argument("--image_size", default=(256,176), type=split, help='Image size')
    parser.add_argument("--input_dir", default='../output/generated_images', help='Folder with images')
    parser.add_argument("--img_index", default=4, type=int,  help='Index of image generated image '
                                                                  'for results with multiple images')
    args = parser.parse_args()
    print (args)

    imgs = []
    for name in os.listdir(args.input_dir):
        img = imread(os.path.join(args.input_dir, name))
        img = img[:, args.img_index * args.image_size[1]:(args.img_index + 1) * args.image_size[1]]
        img = addBounding(img)
        imgs.append(img)

    sc = SSDScorer()
    print (sc.get_score_image_set(imgs))
