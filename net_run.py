# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import numpy as np 
import sys
import scipy
import torch 
import torch.nn as nn 
import time
from scipy import ndimage
from pymic.io.image_read_write import save_nd_array_as_image
from pymic.net_run.agent_seg import  SegmentationAgent
from pymic.net_run.infer_func import Inferer
from pymic.net.net_dict_seg import SegNetDict
from pymic.util.parse_config import parse_config
from pymic.loss.loss_dict_seg import SegLossDict
from net.pfnet import PFNet

local_net_dict = {
    "PFNet": PFNet,
}
local_net_dict.update(SegNetDict)

class SegAgentWithMultiPred(SegmentationAgent):
    def __init__(self, config, stage = 'train'):
        super(SegAgentWithMultiPred, self).__init__(config, stage)
    
    def infer(self):
        device_ids = self.config['testing']['gpus']
        device = torch.device("cuda:{0:}".format(device_ids[0]))
        self.net.to(device)
        # load network parameters and set the network as evaluation mode
        checkpoint_name = self.get_checkpoint_name()
        checkpoint = torch.load(checkpoint_name, map_location = device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        
        if(self.config['testing']['evaluation_mode'] == True):
            self.net.eval()
            if(self.config['testing']['test_time_dropout'] == True):
                def test_time_dropout(m):
                    if(type(m) == nn.Dropout):
                        print('dropout layer')
                        m.train()
                self.net.apply(test_time_dropout)

        infer_cfg = self.config['testing']
        infer_cfg['class_num'] = self.config['network']['class_num']
        infer_obj = Inferer(self.net, infer_cfg)
        infer_time_list = []
        with torch.no_grad():
            for data in self.test_loder:
                images = self.convert_tensor_type(data['image'])
                images = images.to(device)
    
                start_time = time.time()
                pred = infer_obj.run(images)
                # convert tensor to numpy
                if(isinstance(pred, (tuple, list))):
                    # rescale to the same size
                    for i in range(1, len(pred)):
                        pred[i] = nn.functional.interpolate(pred[i], 
                            size = list(pred[0].shape)[2:], mode = 'trilinear')
                    pred = [item.cpu().numpy() for item in pred]
                else:
                    pred = pred.cpu().numpy()
                data['predict'] = pred
                # inverse transform
                for transform in self.transform_list[::-1]:
                    if (transform.inverse):
                        data = transform.inverse_transform_for_prediction(data) 

                infer_time = time.time() - start_time
                infer_time_list.append(infer_time)
                self.save_ouputs(data)
        infer_time_list = np.asarray(infer_time_list)
        time_avg, time_std = infer_time_list.mean(), infer_time_list.std()
        print("testing time {0:} +/- {1:}".format(time_avg, time_std))

    def save_ouputs(self, data):
        output_num = self.config['testing'].get('output_num', 1)
        if(output_num == 1):
            super(SegAgentWithMultiPred, self).save_ouputs(data)
            return 
        print("output_num", output_num)
        output_dir = self.config['testing']['output_dir']
        ignore_dir = self.config['testing'].get('filename_ignore_dir', True)
        label_source = self.config['testing'].get('label_source', None)
        label_target = self.config['testing'].get('label_target', None)
        filename_replace_source = self.config['testing'].get('filename_replace_source', None)
        filename_replace_target = self.config['testing'].get('filename_replace_target', None)
        if(not os.path.exists(output_dir)):
            os.mkdir(output_dir)

        names, pred = data['names'], data['predict']
        # prob = [scipy.special.softmax(item, axis = 1) for item in pred]
        prob = pred
        print("number of prediction",  len(prob))
        print("shape of each prediction")
        for probk in prob:
            print(probk.shape)
        output = np.asarray(np.argmax(prob[0],  axis = 1), np.uint8)
        if((label_source is not None) and (label_target is not None)):
            output = convert_label(output, label_source, label_target)
        # save the output and (optionally) probability predictions
        root_dir  = self.config['dataset']['root_dir']
        for i in range(len(names)):
            save_name = names[i].split('/')[-1] if ignore_dir else \
                names[i].replace('/', '_')
            if((filename_replace_source is  not None) and (filename_replace_target is not None)):
                save_name = save_name.replace(filename_replace_source, filename_replace_target)
            print(save_name)
            save_name = "{0:}/{1:}".format(output_dir, save_name)
            save_nd_array_as_image(output[i], save_name, root_dir + '/' + names[i])
            save_name_split = save_name.split('.')

            # if(not save_prob):
            #     continue
            if('.nii.gz' in save_name):
                save_prefix = '.'.join(save_name_split[:-2])
                save_format = 'nii.gz'
            else:
                save_prefix = '.'.join(save_name_split[:-1])
                save_format = save_name_split[-1]
            
            # save attention maps
            for  k in range(1, len(prob)):
                print("shape of prob k", prob[k].shape)
                pred_k = prob[k][0][1]
                pred_k_savename = "{0:}_att_{1:}.{2:}".format(save_prefix, k, save_format)
                save_nd_array_as_image(pred_k, pred_k_savename, root_dir + '/' + names[i])


def main():
    if(len(sys.argv) < 3):
        print('Number of arguments should be 3. e.g.')
        print('    python net_run_custom.py train config.cfg')
        exit()
    stage    = str(sys.argv[1])
    cfg_file = str(sys.argv[2])
    config   = parse_config(cfg_file)

    agent    = SegAgentWithMultiPred(config, stage)
    net_type = config['network']['net_type']
    net = local_net_dict[net_type](config['network'])
    agent.set_network(net)
    agent.run()

if __name__ == "__main__":
    main()
