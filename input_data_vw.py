# By Jinjun Wu  xnwujj@gmail.com
# Thanks to @Kevin Xu kevin28520@gmail.com
# make a data_list and label_list
import tensorflow as tf
import numpy as np
import os

# you need to change this to your data directory

def get_files(file_dir):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''   
    image_list = []
    label_list = []
    for file in os.listdir(file_dir):
        data_path = file_dir+file
        data_label=int(data_path.split(sep = "class")[1][0:2])-1  #1-10 --> 0-9        print(file_dir+file)
        image_list.append(data_path)
        label_list.append(data_label)
    
    return image_list, label_list
#data_dir = "/home/gps/HDD/dataset_dzkd_radar0612/trains/"
#image_list, label_list=get_files(data_dir)

def get_batch(datadir, image_W, image_H, batch_size, capacity,n_classes):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    image, label=get_files(datadir)   
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)    

    ######################################
    # data argumentation should go to here
    ######################################
    
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)

    # if you want to test the generated batches of images, you might want to comment the following line.
    image = tf.image.per_image_standardization(image)
    
    #you can also use batch
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 1,
                                                capacity = capacity)
    
    #you can also use shuffle_batch 
#    image_batch, label_batch = tf.train.shuffle_batch([image,label],
#                                                      batch_size=BATCH_SIZE,
#                                                      num_threads=64,
#                                                      capacity=CAPACITY,
#                                                      min_after_dequeue=CAPACITY-1)

    ## ONE-HOT      
    label_batch = tf.one_hot(label_batch, depth= n_classes)
    label_batch = tf.cast(label_batch, dtype=tf.int32)
    label_batch = tf.reshape(label_batch, [batch_size, n_classes])
    
    
    ## NO ONE-HOT
#     label_batch = tf.reshape(label_batch, [batch_size])
#     image_batch = tf.cast(image_batch, tf.float32)
    
    return image_batch, label_batch