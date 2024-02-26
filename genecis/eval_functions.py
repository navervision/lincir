# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import clip
from utils.gen_utils import AverageMeter
from utils.metric_utils import get_recall

from transformers import CLIPTextModelWithProjection

from tqdm import tqdm


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    Copy-paste from https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/models/clip/modeling_clip.py#L679-L693
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


@torch.no_grad()
def validate_global(clip_model, combiner, val_image_loader, valloader_global, topk=(1, 5, 10), save_path=None):
    
    combiner.eval()
    clip_model.eval()

    print('Evaluating in GLOBAL setting, extacting features for all images...')
    gallery_ranks = []
    gallery_feats = []
    for batch in tqdm(val_image_loader):

        image, rank = batch
        image = image.cuda()

        image_feature = clip_model.encode_image(image).float()
        image_feature = torch.nn.functional.normalize(image_feature, p=2, dim=-1)

        gallery_feats.append(image_feature)
        gallery_ranks.append(rank.cuda())
        
    gallery_ranks = torch.cat(gallery_ranks)
    gallery_feats = torch.cat(gallery_feats)

    print('Performing eval using Combiner...')
    meters = {k: AverageMeter() for k in topk}
    sims_to_save = []

    with torch.no_grad():
        for batch in tqdm(valloader_global):

            ref_img, caption, ref_global_rank, target_global_rank = [x.cuda(non_blocking=True) for x in batch]
            caption = caption.squeeze()
            if len(target_global_rank.size()) == 1:
                target_global_rank = target_global_rank.unsqueeze(-1) 

            # Forward pass in CLIP
            ref_feats = clip_model.encode_image(ref_img).float()
            caption_feats = clip_model.encode_text(caption).float()

            # Forward pass in combiner
            combined_feats = combiner(ref_feats, caption_feats)
            combined_feats = torch.nn.functional.normalize(combined_feats, p=2, dim=-1)

            # Compute similarity
            similarities = combined_feats[:, None, :] * gallery_feats       # B x N x D
            similarities = similarities.sum(dim=-1)                         # B x N

            # Now mask some similarities (set to -inf) if they correspond the to the same feature in the gallery
            mask = ref_global_rank[:, None].cuda() == gallery_ranks
            similarities[mask] = -1e5

            # Sort the similarities in ascending order (closest example is the predicted sample)
            _, sort_idxs = similarities.sort(dim=-1, descending=True)                   # B x N

            # Compute recall at K
            for k in topk:

                recall_k = get_recall(sort_idxs[:, :k], target_global_rank)
                meters[k].update(recall_k, len(ref_img))

            sims_to_save.append(similarities.cpu())

        # Print results
        print_str = '\n'.join([f'Recall @ {k} = {v.avg:.4f}' for k, v in meters.items()])
        print(print_str)

        if save_path is not None:
            sims_to_save = torch.cat(sims_to_save)
            print(f'Saving text only preds to: {save_path}')
            torch.save(sims_to_save, save_path)
        
        return meters


@torch.no_grad()
def encode_with_pseudo_tokens(clip_model, text: torch.Tensor, pseudo_tokens: torch.Tensor,
                              num_tokens=1) -> torch.Tensor:
    """
    Use the CLIP model to encode a text with pseudo tokens.
    It replaces the word embedding of $ with the pseudo tokens for each element in the batch.
    Based on the original implementation of the CLIP model:
    https://github.com/openai/CLIP/blob/main/clip/model.py
    """
    clip_model = clip_model.module
    x = clip_model.token_embedding(text).type(clip_model.dtype)  # [batch_size, n_ctx, d_model]

    _, counts = torch.unique((text == 259).nonzero(as_tuple=True)[0], return_counts=True)  # 259 is the token of $
    cum_sum = torch.cat((torch.zeros(1, device=text.device).int(), torch.cumsum(counts, dim=0)[:-1]))
    first_tokens_indexes = (text == 259).nonzero()[cum_sum][:, 1]
    rep_idx = torch.cat([(first_tokens_indexes + n).unsqueeze(0) for n in range(num_tokens)])

    if pseudo_tokens.shape[0] == x.shape[0]:
        if len(pseudo_tokens.shape) == 2:
            pseudo_tokens = pseudo_tokens.unsqueeze(1)
        x[torch.arange(x.shape[0]).repeat_interleave(num_tokens).reshape(
            x.shape[0], num_tokens), rep_idx.T] = pseudo_tokens.to(x.dtype)
    else:
        first_tokens_indexes = (text == 259).nonzero()[torch.arange(0, x.shape[0] * num_tokens, num_tokens)][:, 1]
        rep_idx = torch.cat([(first_tokens_indexes + n).unsqueeze(0) for n in range(num_tokens)])
        x[torch.arange(x.shape[0]).repeat_interleave(num_tokens).reshape(
            x.shape[0], num_tokens), rep_idx.T] = pseudo_tokens.repeat(x.shape[0], 1, 1).to(x.dtype)

    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)

    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection

    return x


def encode_with_pseudo_tokens_HF(clip_model: CLIPTextModelWithProjection, text: torch.Tensor, pseudo_tokens: torch.Tensor,
                              num_tokens=1, return_last_states=False) -> torch.Tensor:
    """
    Use the CLIP model to encode a text with pseudo tokens.
    It replaces the word embedding of $ with the pseudo tokens for each element in the batch.
    Based on the original implementation of the CLIP model:
    https://github.com/openai/CLIP/blob/main/clip/model.py
    """
    x = clip_model.text_model.embeddings.token_embedding(text).type(clip_model.dtype)  # [batch_size, n_ctx, d_model]
    x = torch.where(text.unsqueeze(-1) == 259,
                    pseudo_tokens.unsqueeze(1).type(clip_model.dtype),
                    x)
    x = x + clip_model.text_model.embeddings.position_embedding(clip_model.text_model.embeddings.position_ids)
    _causal_attention_mask = _make_causal_mask(text.shape, x.dtype, device=x.device)
    x = clip_model.text_model.encoder(inputs_embeds=x,
                                      attention_mask=None,
                                      causal_attention_mask=_causal_attention_mask,
                                      output_attentions=False,
                                      output_hidden_states=False,
                                      return_dict=False)
    x = x[0]
    x_last = clip_model.text_model.final_layer_norm(x)
    x = x_last[torch.arange(x_last.shape[0], device=x_last.device),
          text.to(dtype=torch.int, device=x_last.device).argmax(dim=-1),
          ]
    if hasattr(clip_model, 'text_projection'):
        x = clip_model.text_projection(x)

    if return_last_states:
        return x, x_last
    else:
        return x


@torch.no_grad()
def validate(image_encoder, text_encoder, combiner, valloader, combiner_mode, topk=(1, 2, 3), save_path=None):
    
    print('Computing eval with combiner...')

    image_encoder.eval()
    text_encoder.eval()
    combiner.eval()

    meters = {k: AverageMeter() for k in topk}
    sims_to_save = []

    with torch.no_grad():
        for batch in tqdm(valloader):

            ref_img, caption, gallery_set, target_rank = [x.cuda(non_blocking=True) for x in batch[:4]]
            string_caption = batch[-1]
            #print(string_caption)
            bsz, n_gallery, _, h, w = gallery_set.size()
            caption = caption.squeeze()

            # Forward pass in CLIP
            imgs_ = torch.cat([ref_img, gallery_set.view(-1, 3, h, w)], dim=0)
            all_img_feats = image_encoder(pixel_values=imgs_).image_embeds#image_encoder.encode_image(imgs_).float()

            # L2 normalize and view into correct shapes
            ref_feats, gallery_feats = all_img_feats.split((bsz, bsz * n_gallery), dim=0)
            gallery_feats = gallery_feats.view(bsz, n_gallery, -1)
            gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=-1)

            # Forward pass in combiner
            if combiner_mode == 'combiner_original':
                caption_feats = text_encoder(input_ids=caption).text_embeds#text_encoder.encode_text(caption).float()
                combined_feats = combiner(ref_feats, caption_feats)
            else:
                # phi, searle
                predicted_tokens = combiner(ref_feats)
                input_captions = [f"a photo of $ that {rel_caption}" for rel_caption in string_caption]
                tokenized_input_captions = clip.tokenize(input_captions, context_length=77).cuda()
                combined_feats = encode_with_pseudo_tokens_HF(text_encoder, tokenized_input_captions, predicted_tokens)

            # Compute similarity
            similarities = combined_feats[:, None, :] * gallery_feats       # B x N x D
            similarities = similarities.sum(dim=-1)                         # B x N

            # Sort the similarities in ascending order (closest example is the predicted sample)
            _, sort_idxs = similarities.sort(dim=-1, descending=True)                   # B x N

            # Compute recall at K
            for k in topk:

                recall_k = get_recall(sort_idxs[:, :k], target_rank)
                meters[k].update(recall_k, bsz)

            sims_to_save.append(similarities.cpu())

        if save_path is not None:
            sims_to_save = torch.cat(sims_to_save)
            print(f'Saving text only preds to: {save_path}')
            torch.save(sims_to_save, save_path)

        # Print results
        print_str = '\n'.join([f'Recall @ {k} = {v.avg:.4f}' for k, v in meters.items()])
        print(print_str)

        return meters
