# Copyright (c) Meta Platforms, Inc. and affiliates.

import os

from datasets.vaw_dataset import VAWValSubset
from datasets.coco_dataset import COCOValSubset

from eval_functions import validate, validate_global
from models.combiner_model import Combiner

import clip
import torch
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPImageProcessor, CLIPTokenizer

from phi import Phi
from model_pic2word import IM2TEXT

import argparse
import torch.backends.cudnn as cudnn
from utils.dist_utils import fix_random_seeds, init_distributed_mode, get_rank
from utils.gen_utils import bool_flag, strip_state_dict, none_flag
from functools import partial

from utils.model_utils import FeatureComb
from utils.dist_utils import CLIPDistDataParallel

#from config import genecis_root

def get_args_parser():

    parser = argparse.ArgumentParser('Eval', add_help=False)

    parser.add_argument('--model', default='RN50x4', type=str, help='Which CLIP model we are using as backbone')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--batch_size_per_gpu', default=8, type=int)

    # Dist params
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    # Model params
    parser.add_argument('--combiner_mode', default='text_only', type=str)
    parser.add_argument('--feature_comb_average', default=0.5, type=float)

    # Pretrain paths
    parser.add_argument('--clip_pretrain_path', default=None, type=none_flag)
    parser.add_argument('--combiner_pretrain_path', default=None, type=none_flag)

    # Dataset params
    parser.add_argument('--coco_val2017_path', default=None, type=str, help='COCO Val 2017 path')
    parser.add_argument('--vg_100k_all_path', default=None, type=str, help='VG 100K path')
    parser.add_argument('--use_complete_text_query', default=False, type=bool_flag, help='Only relevant for MIT States')

    # Save params
    parser.add_argument('--pred_save_path', default=None, type=none_flag, help='Where to save predictions, dont save by default')

    return parser


def build_text_encoder(clip_model_name, cache_dir):
    clip_model_dict = {'base32': 'openai/clip-vit-base-patch32',
                       'base': 'openai/clip-vit-base-patch16',
                       'large': 'openai/clip-vit-large-patch14',
                       'huge': 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
                       'giga': 'Geonmo/CLIP-Giga-config-fixed',
                       'meta-large': 'facebook/metaclip-l14-fullcc2.5b',
                       'meta-huge': 'facebook/metaclip-h14-fullcc2.5b',
                       }

    clip_preprocess = CLIPImageProcessor(crop_size={'height': 224, 'width': 224},
                                         do_center_crop=True,
                                         do_convert_rgb=True,
                                         do_normalize=True,
                                         do_rescale=True,
                                         do_resize=True,
                                         image_mean=[0.48145466, 0.4578275, 0.40821073],
                                         image_std=[0.26862954, 0.26130258, 0.27577711],
                                         resample=3,
                                         size={'shortest_edge': 224},
                                         )

    clip_vision_model = CLIPVisionModelWithProjection.from_pretrained(clip_model_dict[clip_model_name], torch_dtype=torch.float32, cache_dir=cache_dir)

    clip_text_model = CLIPTextModelWithProjection.from_pretrained(clip_model_dict[clip_model_name], torch_dtype=torch.float32, cache_dir=cache_dir)

    tokenizer = CLIPTokenizer.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', subfolder='tokenizer_2', cache_dir=cache_dir)
    #tokenizer.add_special_tokens({'additional_special_tokens':["[$]"]}) # NOTE: 49408

    return clip_vision_model, clip_preprocess, clip_text_model, tokenizer


def main(args):

    # --------------
    # INIT
    # --------------
    fix_random_seeds(0)
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # --------------
    # GET BACKBONE
    # --------------
    print('Loading models...')
    image_encoder, preprocess, text_encoder, tokenizer = build_text_encoder(args.model, '/data/mount_zoo/hf_models')
    feature_dim = text_encoder.config.projection_dim

    # --------------
    # GET COMBINER
    # --------------
    if args.combiner_mode == 'combiner_original':
        combiner = Combiner(clip_feature_dim=feature_dim, projection_dim=2560, hidden_dim=2 * 2560)
    elif args.combiner_mode in ('image_only', 'text_only', 'image_plus_text'):
        combiner = FeatureComb(args.combiner_mode, feature_comb_average=args.feature_comb_average)
    elif args.combiner_mode == 'searle':
        combiner, _ = torch.hub.load(repo_or_dir='miccunifi/SEARLE', model='searle', source='github',
                backbone='ViT-L/14')
    elif args.combiner_mode == 'phi':
        combiner = Phi(input_dim=text_encoder.config.projection_dim,
                       hidden_dim=text_encoder.config.projection_dim * 4,
                       output_dim=text_encoder.config.hidden_size, dropout=0.5)
        combiner.eval()
    elif args.combiner_mode == 'pic2word':
        combiner = IM2TEXT(embed_dim=text_encoder.config.projection_dim, output_dim=text_encoder.config.hidden_size)
    else:
        raise ValueError

    # --------------
    # LOAD PRETRAINED WEIGHTS
    # --------------
    if args.combiner_pretrain_path is not None:
        state_dict = torch.load(args.combiner_pretrain_path, map_location='cpu')
        if args.combiner_mode == 'phi':
            combiner.load_state_dict(state_dict[combiner.__class__.__name__])
        elif args.combiner_mode == 'pic2word':
            state_dict = state_dict['state_dict_img2text']
            state_dict = strip_state_dict(state_dict=state_dict, strip_key='module.')
            combiner.load_state_dict(state_dict)

    # --------------
    # To cuda
    # --------------
    image_encoder, text_encoder, combiner = image_encoder.cuda(), text_encoder.cuda(), combiner.cuda()

    # --------------
    # GET DATASET
    # --------------
    print('Loading datasets...')
    tokenizer = partial(clip.tokenize, truncate=True)
    
    dataset_list = ['focus_attribute', 'change_attribute', 'focus_object', 'change_object']
    genecis_root = './genecis'
    for dataset in dataset_list:
        genecis_split_path = os.path.join(genecis_root, f'{dataset}.json')
        if 'attribute' in dataset:
            print(f'Evaluating on GeneCIS from {genecis_split_path}')

            val_dataset_subset = VAWValSubset(image_dir=args.vg_100k_all_path, val_split_path=genecis_split_path, tokenizer=tokenizer, transform=preprocess)
            print(f'Evaluating on {len(val_dataset_subset)} templates...')

        elif 'object' in dataset:
            print(f'Evaluating on GeneCIS from {genecis_split_path}')

            val_dataset_subset = COCOValSubset(root_dir=args.coco_val2017_path, val_split_path=genecis_split_path, tokenizer=tokenizer, transform=preprocess)
            print(f'Evaluating on {len(val_dataset_subset)} templates...')

        # --------------
        # GET DATALOADER
        # --------------
        get_dataloader = partial(torch.utils.data.DataLoader, sampler=None,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False)

        valloader_subset = get_dataloader(dataset=val_dataset_subset)
        validate(image_encoder, text_encoder, combiner, valloader_subset, args.combiner_mode, topk=(1, 2, 3), save_path=args.pred_save_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Eval', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
    
