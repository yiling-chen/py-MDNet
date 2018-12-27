from __future__ import print_function
import numpy as np
import os
import cv2
import sys
import time
import random
import argparse
import json
from PIL import Image
import matplotlib.pyplot as plt
from os.path import join
from glob import glob
from os import walk

import torch
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable

sys.path.insert(0, '../modules')
from sample_generator import *
from data_prov import *
from model import *
from bbreg import *
from options import *
from gen_config import *

np.random.seed(123)
torch.manual_seed(456)
torch.cuda.manual_seed(789)

def forward_samples(model, image, samples, out_layer='conv3'):
    model.eval()
    extractor = RegionExtractor(image, samples, opts['img_size'], opts['padding'], opts['batch_test'])
    for i, regions in enumerate(extractor):
        regions = Variable(regions)
        if opts['use_gpu']:
            regions = regions.cuda()
        feat = model(regions, out_layer=out_layer)
        if i==0:
            feats = feat.data.clone()
        else:
            feats = torch.cat((feats,feat.data.clone()),0)
    return feats


def set_optimizer(model, lr_base, lr_mult=opts['lr_mult'], momentum=opts['momentum'], w_decay=opts['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.iteritems():
        lr = lr_base
        for l, m in lr_mult.iteritems():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr':lr})
    optimizer = optim.SGD(param_list, lr = lr, momentum=momentum, weight_decay=w_decay)
    return optimizer


def train(model, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4'):
    model.train()
    
    batch_pos = opts['batch_pos']
    batch_neg = opts['batch_neg']
    batch_test = opts['batch_test']
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)

    pos_idx = np.random.permutation(pos_feats.size(0))
    neg_idx = np.random.permutation(neg_feats.size(0))
    while(len(pos_idx) < batch_pos*maxiter):
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
    while(len(neg_idx) < batch_neg_cand*maxiter):
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
    pos_pointer = 0
    neg_pointer = 0

    for iter in range(maxiter):

        # select pos idx
        pos_next = pos_pointer+batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer+batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_feats = Variable(pos_feats.index_select(0, pos_cur_idx))
        batch_neg_feats = Variable(neg_feats.index_select(0, neg_cur_idx))

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()
            for start in range(0,batch_neg_cand,batch_test):
                end = min(start+batch_test,batch_neg_cand)
                score = model(batch_neg_feats[start:end], in_layer=in_layer)
                if start==0:
                    neg_cand_score = score.data[:,1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.data[:,1].clone()),0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats.index_select(0, Variable(top_idx))
            model.train()
        
        # forward
        pos_score = model(batch_pos_feats, in_layer=in_layer)
        neg_score = model(batch_neg_feats, in_layer=in_layer)
        
        # optimize
        loss = criterion(pos_score, neg_score)
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
        optimizer.step()

        #print "Iter %d, Loss %.4f" % (iter, loss.data[0])


def run_mdnet():
    sequence_root_folder = 'sequence-3'

    files = glob(join(sequence_root_folder, '*.json'))
    print(files[0])

    with open(files[0]) as f:
        data = f.read()

    json_obj = json.loads(data)

    # read and cache all frames
    img_list = glob(join(sequence_root_folder, '*.jpg'))
    img_list.sort()

    src_images = []
    frames = []
    for f in img_list:
        src_images.append(Image.open(f).convert('RGB'))
        frames.append(cv2.imread(f))

    height, width = frames[0].shape[:2]

    # Init model
    model = MDNet(opts['model_path'])
    if opts['use_gpu']:
        model = model.cuda()
    model.set_learnable_params(opts['ft_layers'])
    
    # Init criterion and optimizer 
    criterion = BinaryLoss()
    init_optimizer = set_optimizer(model, opts['lr_init'])
    update_optimizer = set_optimizer(model, opts['lr_update'])

    for obj_id in range(len(json_obj['annotations'][0]['data']['objects'])):
        # get the first frame
        frm = json_obj['annotations'][0]['data']['objects'][obj_id]['frames'][0]
        x1 = int(frm['boundingBox']['x1'] * width)
        y1 = int(frm['boundingBox']['y1'] * height)
        x2 = int(frm['boundingBox']['x2'] * width)
        y2 = int(frm['boundingBox']['y2'] * height)

        frm_idx = frm['frameNumber']
        cv2.rectangle(frames[frm_idx], (x1, y1), (x2, y2), (0,0,255), 2)

        # Init bbox
        num_frames = len(json_obj['annotations'][0]['data']['objects'][obj_id]['frames'])

        target_bbox = np.array([x1, y1, x2-x1, y2-y1])
        result = np.zeros((num_frames,4))
        result_bb = np.zeros((num_frames,4))
        result[0] = target_bbox
        result_bb[0] = target_bbox

        tic = time.time()
        # Load first image
        # image = Image.open(img_list[0]).convert('RGB')
        image = src_images[frm_idx]
        
        # Train bbox regressor
        bbreg_examples = gen_samples(SampleGenerator('uniform', image.size, 0.3, 1.5, 1.1),
                                    target_bbox, opts['n_bbreg'], opts['overlap_bbreg'], opts['scale_bbreg'])
        bbreg_feats = forward_samples(model, image, bbreg_examples)
        bbreg = BBRegressor(image.size)
        bbreg.train(bbreg_feats, bbreg_examples, target_bbox)

        # Draw pos/neg samples
        pos_examples = gen_samples(SampleGenerator('gaussian', image.size, 0.1, 1.2),
                                target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

        neg_examples = np.concatenate([
                        gen_samples(SampleGenerator('uniform', image.size, 1, 2, 1.1), 
                                    target_bbox, opts['n_neg_init']//2, opts['overlap_neg_init']),
                        gen_samples(SampleGenerator('whole', image.size, 0, 1.2, 1.1),
                                    target_bbox, opts['n_neg_init']//2, opts['overlap_neg_init'])])
        neg_examples = np.random.permutation(neg_examples)

        # Extract pos/neg features
        pos_feats = forward_samples(model, image, pos_examples)
        neg_feats = forward_samples(model, image, neg_examples)
        feat_dim = pos_feats.size(-1)

        # Initial training
        train(model, criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'])
        
        # Init sample generators
        sample_generator = SampleGenerator('gaussian', image.size, opts['trans_f'], opts['scale_f'], valid=True)
        pos_generator = SampleGenerator('gaussian', image.size, 0.1, 1.2)
        neg_generator = SampleGenerator('uniform', image.size, 1.5, 1.2)

        # Init pos/neg features for update
        pos_feats_all = [pos_feats[:opts['n_pos_update']]]
        neg_feats_all = [neg_feats[:opts['n_neg_update']]]

        # Main loop
        color = (random.randint(1, 255), random.randint(1, 255), random.randint(1, 128))
        for j in range(1, num_frames):

            frm = json_obj['annotations'][0]['data']['objects'][obj_id]['frames'][j]
            frm_idx = frm['frameNumber']
            x1 = int(frm['boundingBox']['x1'] * width)
            y1 = int(frm['boundingBox']['y1'] * height)
            x2 = int(frm['boundingBox']['x2'] * width)
            y2 = int(frm['boundingBox']['y2'] * height)
            
            # draw ground truth
            cv2.rectangle(frames[frm_idx], (x1, y1), (x2, y2), (0,0,255), 2)

            timer = cv2.getTickCount()

            # Load image
            image = src_images[frm_idx]

            # Estimate target bbox
            samples = gen_samples(sample_generator, target_bbox, opts['n_samples'])
            sample_scores = forward_samples(model, image, samples, out_layer='fc6')
            top_scores, top_idx = sample_scores[:,1].topk(5)
            top_idx = top_idx.cpu().numpy()
            target_score = top_scores.mean()
            target_bbox = samples[top_idx].mean(axis=0)

            success = target_score > opts['success_thr']
            
            # Expand search area at failure
            if success:
                sample_generator.set_trans_f(opts['trans_f'])
            else:
                sample_generator.set_trans_f(opts['trans_f_expand'])

            # Bbox regression
            if success:
                bbreg_samples = samples[top_idx]
                bbreg_feats = forward_samples(model, image, bbreg_samples)
                bbreg_samples = bbreg.predict(bbreg_feats, bbreg_samples)
                bbreg_bbox = bbreg_samples.mean(axis=0)
            else:
                bbreg_bbox = target_bbox
            
            # Copy previous result at failure
            if not success:
                target_bbox = result[j-1]
                bbreg_bbox = result_bb[j-1]
            
            # Save result
            result[j] = target_bbox
            result_bb[j] = bbreg_bbox

            # Data collect
            if success:
                # Draw pos/neg samples
                pos_examples = gen_samples(pos_generator, target_bbox, 
                                        opts['n_pos_update'],
                                        opts['overlap_pos_update'])
                neg_examples = gen_samples(neg_generator, target_bbox, 
                                        opts['n_neg_update'],
                                        opts['overlap_neg_update'])

                # Extract pos/neg features
                pos_feats = forward_samples(model, image, pos_examples)
                neg_feats = forward_samples(model, image, neg_examples)
                pos_feats_all.append(pos_feats)
                neg_feats_all.append(neg_feats)
                if len(pos_feats_all) > opts['n_frames_long']:
                    del pos_feats_all[0]
                if len(neg_feats_all) > opts['n_frames_short']:
                    del neg_feats_all[0]

            # Short term update
            if not success:
                nframes = min(opts['n_frames_short'],len(pos_feats_all))
                pos_data = torch.stack(pos_feats_all[-nframes:],0).view(-1,feat_dim)
                neg_data = torch.stack(neg_feats_all,0).view(-1,feat_dim)
                train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])
            
            # Long term update
            elif j % opts['long_interval'] == 0:
                pos_data = torch.stack(pos_feats_all,0).view(-1,feat_dim)
                neg_data = torch.stack(neg_feats_all,0).view(-1,feat_dim)
                train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])
            
            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            bbox = [int(x) for x in bbreg_bbox]
            cv2.rectangle(frames[frm_idx], (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), color, 2)

            # Display tracker type on frame
            cv2.putText(frames[frm_idx], "MDNet", (50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230,170,50),2)

            # Display FPS on frame
            # cv2.putText(frames[frm_idx], "FPS : " + str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (230,170,50), 2)


    writer = cv2.VideoWriter(sequence_root_folder + '-MDNet.avi', cv2.VideoWriter_fourcc('X','V','I','D'), 15, (width, height))

    if not writer.isOpened():
        print('Failed to open video')
        return

    for frm in frames:
        writer.write(frm)

    writer.release()
    return


run_mdnet()