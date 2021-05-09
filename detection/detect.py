import argparse
import os
from PIL import Image
import numpy as np
import copy
from glob import glob
import random
import cv2
import time
import multiprocessing as mp
from functools import partial
import darknet
import darknet_images

def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    assert 0 < args.resolution, "Resolution should be an integer greater than zero"
    if not os.path.exists(args.input_folder):
        raise(ValueError("Invalid input folder path {}".format(os.path.abspath(args.input_folder))))
    if not os.path.exists(args.output_folder):
        raise(ValueError("Invalid output folder path {}".format(os.path.abspath(args.output_folder))))
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))

def check_batch_shape(images, batch_size):  
    """
        Image sizes should be the same width and height
    """
    shapes = [image.shape for image in images]
    if len(set(shapes)) > 1:
        raise ValueError("Images don't have same shape")
    if len(shapes) > batch_size:
        raise ValueError("Batch size higher than number of images")
    return shapes[0]

def get_images(input_folder):

    images = []

    for ext in ('*.jpg', '*.jpeg', '*.png'):
        images.extend(glob(input_folder + ext))
    
    print('{} image files recognised in the provided input folder.'.format(len(images)))

    return images

def split_image(img_path, resolution):

    # Get image
    image = np.asarray(Image.open(img_path))
    w = image.shape[0]
    h = image.shape[1]

    # Store cropped images' names
    img_crop_names = []

    # Check if input resolution is high enough
    if w < args.resolution or h < args.resolution:
        print('Image |{}| resolution too low: {}x{}'.format(img_path, w, h))
        print('Please provide images with resolution at least {}x{}'.format(args.resolution, args.resolution))
        exit()

    # Determine number of sub-images to create
    nwi = int(np.ceil(w/args.resolution))
    wo = int(np.ceil((nwi*args.resolution - w)/(nwi - 1)))
    nhi = int(np.ceil(h/args.resolution))
    ho = int(np.ceil((nhi*args.resolution - h)/(nhi - 1)))

    # Generate crop images
    for i in range(nwi):
        for j in range(nhi):
            if i == nwi-1:
                i_s = w - args.resolution
                i_e = w
            else:
                i_s = i*args.resolution - i*wo
                i_e = i_s + args.resolution
            if j == nhi -1:
                j_s = h - args.resolution
                j_e = h
            else: 
                j_s = j*args.resolution - j*ho
                j_e = j_s + args.resolution

            snip = image[i_s:i_e, j_s:j_e]
            Image.fromarray(snip).save(args.output_folder + os.path.splitext(os.path.basename(img_path))[0] + '_' + str(j_s) + '_' + str(i_s) + '.jpg')
            img_crop_names.append(args.output_folder + os.path.splitext(os.path.basename(img_path))[0] + '_' + str(j_s) + '_' + str(i_s) + '.jpg')

    return img_crop_names

def run_detection(output_splits, args):
    # Determine an optimal batch size
    if args.batch_size == None:
        args.batch_size = 1
        n_images = len(output_splits)
        while(n_images%(2*args.batch_size) == 0):
            args.batch_size *= 2
        n_batches = int(n_images/args.batch_size)

    print('Optimal batch size: {}'.format(args.batch_size))
    print('Loading network.')

    # Load network
    random.seed(3) # Deterministic bbox colours
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=args.batch_size
    )
    
    # Get network size
    n_w = darknet.network_width(network)
    n_h = darknet.network_height(network)

    # Split in batches
    global_predictions = []
    drawn_images = []
    for batch in range(n_batches):

        prev_time = time.time() # Get batch calculation start time

        # Get images in the batch
        names_batch = output_splits[batch*args.batch_size:((batch+1)*args.batch_size)]
        images_batch = [cv2.imread(n) for n in names_batch] 
        images_batch = [cv2.resize(n, (n_w, n_h), interpolation=cv2.INTER_LINEAR) for n in images_batch] # Ensure image resolution is the same as network's
        h, w, _ = check_batch_shape(images_batch, args.batch_size)
        dnet_images = darknet_images.prepare_batch(images_batch, network)

        # Detect
        batch_detections = darknet.network_predict_batch(
            network,
            dnet_images,
            args.batch_size,
            w,
            h,
            args.thresh,
            args.hier_thresh,
            None,
            0,
            0
        )

        # Process detections
        batch_predictions = []
        for idx in range(args.batch_size):
            num = batch_detections[idx].num
            detections = batch_detections[idx].dets
            if args.nms:
                darknet.do_nms_obj(detections, num, len(class_names), args.nms)
            predictions = darknet.remove_negatives(detections, class_names, num)
            images_batch[idx] = darknet.draw_boxes(predictions, images_batch[idx], class_colors)
            batch_predictions.append(predictions)
        darknet.free_batch_detections(batch_detections, args.batch_size)

        # Store predictions and drawn images
        global_predictions.extend(batch_predictions)
        images_batch = [cv2.cvtColor(n, cv2.COLOR_BGR2RGB) for n in images_batch] # Ensure output is in RGB
        drawn_images.extend(images_batch)

        # Calculate FPS
        fps = int(args.batch_size/(time.time() - prev_time))
        print("FPS: {}".format(fps))

    return drawn_images, global_predictions, class_names


def main():
    
    # Get start time
    start_time = time.time()
    
    # Check argument errors
    check_arguments_errors(args)

    # Load data
    images = get_images(input_folder=args.input_folder)

    # Split images
    # TODO: Need to paralelize
    output_splits = []   
    for i in images:

        # Split image
        output_splits.extend(split_image(i, args.resolution))

    print('{} output splits successfully genrated.'.format(len(output_splits)))
    print('Running detection.')

    # Run detection
    output_images, predictions, class_names = run_detection(output_splits, args)

    # Display and save results
    for img, pred, name in zip(output_images, predictions, output_splits):
        darknet_images.save_annotations(name, img, pred, class_names)
        # darknet.print_detections(pred, args.ext_output)
        
    # Output average processing time per image
    avg_proc_time = (time.time() - start_time)/len(images)
    print("Average processing time per image: {}".format(avg_proc_time))

if __name__ == "__main__":

    # TODO: Set defaults to automate servce/client
    # Define parser for input arguments
    parser = argparse.ArgumentParser(description='Split images in smaller ones')
    parser.add_argument('--input_folder', '-if', type=str)
    parser.add_argument('--output_folder', '-of', type=str)
    parser.add_argument('--resolution', '-res', type=int, default=720)
    parser.add_argument('--config_file', '-cfg', type=str)
    parser.add_argument('--data_file', '-df', type=str)
    parser.add_argument('--weights', '-w', type=str)
    parser.add_argument('--batch_size', '-bs', type=int)
    parser.add_argument('--thresh', '-t', type=float, default=0.25)
    parser.add_argument('--hier_thresh', '-ht', type=float, default=0.5)
    parser.add_argument('--nms', '-nms', type=float, default=0.45)
    parser.add_argument('--ext_output', '-ext', type=bool, default=True)
    args = parser.parse_args()

    main()