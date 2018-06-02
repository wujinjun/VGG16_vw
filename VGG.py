
import tensorflow as tf
import tools

def VGG16N(x, n_classes, is_pretrain=True):
    
    with tf.name_scope('VGG16'):

        conv1_1 = tools.conv('conv1_1', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)   
        conv1_2 = tools.conv('conv1_2', conv1_1, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        with tf.name_scope('pool1'):    
            pool1 = tools.pool('pool1', conv1_2, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
            
        conv2_1 = tools.conv('conv2_1', pool1, 128, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)    
        conv2_2 = tools.conv('conv2_2', conv2_1, 128, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        with tf.name_scope('pool2'):    
            pool2 = tools.pool('pool2', conv2_2, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
         
            

        conv3_1 = tools.conv('conv3_1', pool2, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        conv3_2 = tools.conv('conv3_2', conv3_1, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        conv3_3 = tools.conv('conv3_3', conv3_2, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        with tf.name_scope('pool3'):
            pool3 = tools.pool('pool3', conv3_3, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
            

        conv4_1 = tools.conv('conv4_1', pool3, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        conv4_2 = tools.conv('conv4_2', conv4_1, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        conv4_3 = tools.conv('conv4_3', conv4_2, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        with tf.name_scope('pool4'):
            pool4 = tools.pool('pool4', conv4_3, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
        

        conv5_1 = tools.conv('conv5_1', pool4, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        conv5_2 = tools.conv('conv5_2', conv5_1, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        conv5_3 = tools.conv('conv5_3', conv5_2, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        with tf.name_scope('pool5'):
            pool5 = tools.pool('pool5', conv5_3, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)            
        
        
        fc6 = tools.FC_layer('fc6', pool5, out_nodes=4096)        
        with tf.name_scope('batch_norm1'):
            batch_norm1 = tools.batch_norm(fc6)           
        fc7 = tools.FC_layer('fc7', batch_norm1, out_nodes=4096)        
        with tf.name_scope('batch_norm2'):
            batch_norm2 = tools.batch_norm(fc7)            
        fc8 = tools.FC_layer('fc8', batch_norm2, out_nodes=n_classes)
    
        return fc8
        #return conv1_1,conv1_2,conv2_1,conv2_2,conv3_1,conv3_2,conv3_3,conv4_1,conv4_2,conv4_3,conv5_1,conv5_2,conv5_3,pool1,pool2,pool3,pool4,pool5,fc8
        #return conv3_1,conv3_2,conv3_3,fc8

def VGG16N_fc6(x, n_classes, is_pretrain=True):
    
    with tf.name_scope('VGG16'):

        conv1_1 = tools.conv('conv1_1', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)   
        conv1_2 = tools.conv('conv1_2', conv1_1, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        with tf.name_scope('pool1'):    
            pool1 = tools.pool('pool1', conv1_2, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
            
        conv2_1 = tools.conv('conv2_1', pool1, 128, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)    
        conv2_2 = tools.conv('conv2_2', conv2_1, 128, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        with tf.name_scope('pool2'):    
            pool2 = tools.pool('pool2', conv2_2, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
         
            

        conv3_1 = tools.conv('conv3_1', pool2, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        conv3_2 = tools.conv('conv3_2', conv3_1, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        conv3_3 = tools.conv('conv3_3', conv3_2, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        with tf.name_scope('pool3'):
            pool3 = tools.pool('pool3', conv3_3, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
            

        conv4_1 = tools.conv('conv4_1', pool3, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        conv4_2 = tools.conv('conv4_2', conv4_1, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        conv4_3 = tools.conv('conv4_3', conv4_2, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        with tf.name_scope('pool4'):
            pool4 = tools.pool('pool4', conv4_3, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
        

        conv5_1 = tools.conv('conv5_1', pool4, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        conv5_2 = tools.conv('conv5_2', conv5_1, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        conv5_3 = tools.conv('conv5_3', conv5_2, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        with tf.name_scope('pool5'):
            pool5 = tools.pool('pool5', conv5_3, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)            
        
        
        fc6 = tools.FC_layer('fc6', pool5, out_nodes=4096)        
        with tf.name_scope('batch_norm1'):
            batch_norm1 = tools.batch_norm(fc6)           
        fc7 = tools.FC_layer('fc7', batch_norm1, out_nodes=4096)        
        with tf.name_scope('batch_norm2'):
            batch_norm2 = tools.batch_norm(fc7)            
        fc8 = tools.FC_layer('fc8', batch_norm2, out_nodes=n_classes)
    
        return fc6






            