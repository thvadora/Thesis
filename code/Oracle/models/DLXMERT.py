from types import SimpleNamespace
import copy

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.LXMERTOracleInputTarget import LXMERTOracleInputTarget as LXMERT
from utils.model_loading import load_model

use_cuda = torch.cuda.is_available()

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class DLXMERT(nn.Module):

    def __init__(self, lxmert_oracle_dict, lxmert_oracle_weights, pretrain_size):
        super().__init__()
        self.lxmert = LXMERT(
            no_words            = lxmert_oracle_dict['no_words'],
            no_words_feat       = lxmert_oracle_dict['no_words_feat'],
            no_categories       = lxmert_oracle_dict['no_categories'],
            no_category_feat    = lxmert_oracle_dict['no_category_feat'],
            no_hidden_encoder   = lxmert_oracle_dict['no_hidden_encoder'],
            mlp_layer_sizes     = lxmert_oracle_dict['mlp_layer_sizes'],
            no_visual_feat      = lxmert_oracle_dict['no_visual_feat'],
            no_crop_feat        = lxmert_oracle_dict['no_crop_feat'],
            dropout             = lxmert_oracle_dict['dropout'],
            inputs_config       = lxmert_oracle_dict['inputs_config'],
            scale_visual_to     = lxmert_oracle_dict['scale_visual_to'],
            lxmert_encoder_args = lxmert_oracle_dict['lxmert_encoder_args']
        )
        self.lxmert = load_model(self.lxmert, lxmert_oracle_weights, use_dataparallel=use_cuda)
        
        #encoder only
        self.lxmert_encoder = copy.deepcopy(self.lxmert)
        self.lxmert_encoder.module.mlp = Identity()
        self.lxmert_encoder.eval()

        #extract crossAtt
        self.extractions = {}
        if pretrain_size == 'small':
            self.lxmert_encoder.module.lxrt_encoder.model.bert.encoder.x_layers[1].visual_attention.output.dense.register_forward_hook(self.extract('crossAtt'))
   
    def extract(self, name):
        def hook(model, input, output):
            self.extractions[name] = output.detach()
        return hook

    def forward(self, questions, obj_categories, spatials, crop_features, visual_features, lengths,
                history_raw, fasterrcnn_features, fasterrcnn_boxes, target_bbox):
        lxmert_out = self.lxmert_encoder(questions,
                obj_categories,
                spatials,
                crop_features,
                visual_features,
                lengths,
                history_raw,
                fasterrcnn_features,
                fasterrcnn_boxes,
                target_bbox
            )
        crossAtt = self.extractions['crossAtt']
        return (lxmert_out, crossAtt)