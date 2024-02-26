# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch

class FeatureComb(torch.nn.Module):

    def __init__(self, mode, feature_comb_average=0.5) -> None:
        super().__init__()
        assert mode in ('image_only', 'text_only', 'image_plus_text')
        self.mode = mode
        self.feature_comb_average = feature_comb_average

    def forward(self, image_feats, caption_feats):

        if self.mode == 'image_plus_text':

            combined_feat = self.feature_comb_average * image_feats + (1 - self.feature_comb_average) * caption_feats
        
        elif self.mode == 'image_only':
            combined_feat = image_feats
        
        elif self.mode == 'text_only':
            combined_feat = caption_feats
        
        combined_feat = torch.nn.functional.normalize(combined_feat, dim=-1)

        return combined_feat

