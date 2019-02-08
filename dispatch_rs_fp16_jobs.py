#!/bin/python3
# Yi He
# University of Chicago
#####################################################################
# This file intends to generate all testing instances for ResNet FP16 filtered image delta propagation
# We only run per-layer bound and top1_top3 techniques for now    

import psycopg2
import numpy as np
import random

# For ResNet layers, we first select the starting and ending convolution layes that are separate from residual blocks,
# Then from each residual block, we include its shortcut layer, which is one for each block,
# Then for each residual block, we select one shortcut unit
# For the first block, we select the first unit: block1/unit_1/conv1-3
# For the second block, we select the fourth unit: block2/unit_4/conv1-3
# For the third block, we select the fifth unit: block3/unit_5/conv1-3
# For the fourth block, we select the second unit: block4/unit_2/conv1-3
layer_list = ['resnet_model/conv2d/Conv2D','resnet_model/conv2d_2/Conv2D','resnet_model/conv2d_3/Conv2D','resnet_model/conv2d_4/Conv2D','resnet_model/conv2d_20/Conv2D','resnet_model/conv2d_21/Conv2D','resnet_model/conv2d_22/Conv2D','resnet_model/conv2d_43/Conv2D','resnet_model/conv2d_44/Conv2D','resnet_model/conv2d_45/Conv2D','resnet_model/conv2d_50/Conv2D','resnet_model/conv2d_51/Conv2D','resnet_model/conv2d_52/Conv2D']

try:
    conn = psycopg2.connect(database='neuron_inject', user='yiizy', password='21stCenszMan!', host='yanjing-compute1.cs.uchicago.edu')
except:
    print('Unable to connect to the database.')

# Get All inject images and their golden label (The network's golden label, which is not necessarily the image's true label)
cur = conn.cursor()
cur.execute("""SELECT DISTINCT images.image_id, rs5_fp16_golden_label FROM images WHERE image_id < 401;""")
results = cur.fetchall()

for elem in results:
    # Get image ID
    image_id = elem[0]

    # Get the top1 and top2 labels
    top1 = elem[1][0]
    top2 = elem[1][1]
    # For all layers
    for layer in layer_list:
        # Retrieve the target neuron from saliency database
        target_h = -1
        target_w = -1
        target_c = -1
        cur.execute("""SELECT rs10h[1], rs10w[1], rs10c[1] FROM rs_fp16_saliency WHERE image_id = """ + str(image_id) + """ AND layer = \'""" + layer + """\'""")
        result = cur.fetchall()[0]
        target_h = result[0]
        target_w = result[1]
        target_c = result[2]
        
        #for bound_type in ['LY','TP13']:
        for bound_type in ['LY','TP13']:
            exe_str = """INSERT INTO rs_new_fp16_filtered_DP (job_status, image_id, layer, bound_type, inj_h, inj_w, inj_c, golden_label, comparing_label, delta, loss, loss_diff, fc3_golden_label, fc3_comparing_label, top5_label, top5_prob) VALUES ('READY',""" + str(image_id) + """,\'""" + layer + """\',\'""" + bound_type + """\',""" + str(target_h) + """,""" + str(target_w) + """,""" + str(target_c) + """,""" + str(top1) + """,""" + str(top2) + """,NULL,NULL,NULL,NULL,NULL,NULL,NULL);"""
            #print(exe_str)
            cur.execute(exe_str)
            conn.commit()
