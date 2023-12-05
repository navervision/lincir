'''
LinCIR
Copyright (c) 2023-present NAVER Corp.
CC BY-NC-4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
'''
import os
import time
from argparse import ArgumentParser

import numpy as np
import torch
import gradio as gr
from clip_retrieval.clip_client import ClipClient

from encode_with_pseudo_tokens import encode_with_pseudo_tokens_HF
from models import build_text_encoder, Phi, PIC2WORD

import transformers
from transformers.models.clip.modeling_clip import _make_causal_mask
from diffusers import StableDiffusionPipeline #StableDiffusionXLPipeline


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", default="trained_models", type=str,
                        help="The output directory where the model predictions and checkpoints will be written")
    parser.add_argument("--pic2word_ckpt_path", default="trained_models", type=str)
    parser.add_argument("--cache_dir", default="./hf_models", type=str,
                        help="Path to model cache folder")
    parser.add_argument("--clip_model_name", default="giga", type=str,
                        help="CLIP model to use, e.g 'large', 'giga'")
    parser.add_argument("--mixed_precision", default="fp16", type=str)
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--test_fps", action="store_true")
    args = parser.parse_args()
    return args


def load_models(args):
    clip_vision_model, clip_preprocess, clip_text_model, tokenizer = build_text_encoder(args)

    tokenizer.add_special_tokens({'additional_special_tokens':["$1", "$2"]}) # 49409, 49410

    # ours
    phi = Phi(input_dim=clip_text_model.config.projection_dim,
              hidden_dim=clip_text_model.config.projection_dim * 4,
              output_dim=clip_text_model.config.hidden_size, dropout=0.0)
    phi.eval()

    # searle
    phi_searle, _ = torch.hub.load(repo_or_dir='miccunifi/SEARLE', model='searle', source='github',
                                   backbone='ViT-L/14')
    phi_searle.eval()

    # pic2word
    phi_pic2word = PIC2WORD(embed_dim=clip_text_model.config.projection_dim,
                            output_dim=clip_text_model.config.hidden_size)
    phi_pic2word.eval()

    device, dtype = 'cuda:0', torch.float16

    clip_vision_model.to(device, dtype=dtype)
    clip_text_model.to(device, dtype=dtype)

    if not args.test_fps:
        state_dict = torch.load(args.ckpt_path)
        phi.load_state_dict(state_dict['Phi'])

        phi_searle.to(device, dtype=dtype)

        sd = torch.load(args.pic2word_ckpt_path, map_location=device)['state_dict_img2text']
        sd = {k[len('module.'):]: v for k, v in sd.items()}
        phi_pic2word.load_state_dict(sd)
    phi.to(device, dtype=dtype)
    phi_pic2word.to(device, dtype=dtype)

    #decoder = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1', torch_dtype=dtype, variant='fp16', use_safetensors=True)
    #decoder.to(device)
    decoder = None

    return {'clip_vision_model': clip_vision_model,
            'clip_preprocess': clip_preprocess,
            'clip_text_model': clip_text_model,
            'tokenizer': tokenizer,
            'phi': phi,
            'phi_searle': phi_searle,
            'phi_pic2word': phi_pic2word,
            'decoder': decoder,
            'device': device,
            'dtype': dtype,
            }



def encode_with_pseudo_tokens_HF_test(clip_model, text: torch.Tensor, pseudo_tokens1: torch.Tensor, pseudo_tokens2,
                              num_tokens=1, return_last_states=False) -> torch.Tensor:
    """
    Use the CLIP model to encode a text with pseudo tokens.
    It replaces the word embedding of $ with the pseudo tokens for each element in the batch.
    Based on the original implementation of the CLIP model:
    https://github.com/openai/CLIP/blob/main/clip/model.py
    """
    x = clip_model.text_model.embeddings.token_embedding(text).type(clip_model.dtype)  # [batch_size, n_ctx, d_model]
    x = torch.where(text.unsqueeze(-1) == 259,
                    pseudo_tokens1.unsqueeze(1).type(clip_model.dtype),
                    x)
    if pseudo_tokens2 is not None:
        x = torch.where(text.unsqueeze(-1) == 260,
                        pseudo_tokens2.unsqueeze(1).type(clip_model.dtype),
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

def predict(images, images2, input_text, do_generate, model_name):
    if images2:
        input_images = model_dict['clip_preprocess']([images, images2], return_tensors='pt')['pixel_values'].to(model_dict['device'])
    else:
        input_images = model_dict['clip_preprocess'](images, return_tensors='pt')['pixel_values'].to(model_dict['device'])
        input_text = input_text.replace('$2', '')

    input_tokens = model_dict['tokenizer'](text=input_text, return_tensors='pt', padding='max_length', truncation=True)['input_ids'].to(model_dict['device'])
    input_tokens = torch.where(input_tokens == 49409,
                               torch.ones_like(input_tokens) * 259,
                               input_tokens)
    input_tokens = torch.where(input_tokens == 49410,
                               torch.ones_like(input_tokens) * 260,
                               input_tokens)
    image_features = model_dict['clip_vision_model'](pixel_values=input_images.half()).image_embeds
    if model_name == 'ours':
        estimated_token_embeddings = model_dict['phi'](image_features)
    elif model_name == 'searle':
        estimated_token_embeddings = model_dict['phi_searle'](image_features)
    else: # model_name == 'pic2word'
        estimated_token_embeddings = model_dict['phi_pic2word'](image_features)
    if images2:
        estimated_token_embeddings, estimated_token_embeddings2 = estimated_token_embeddings.chunk(2)
    else:
        estimated_token_embeddings2 = None
    text_embeddings, text_last_hidden_states = encode_with_pseudo_tokens_HF_test(model_dict['clip_text_model'], input_tokens, estimated_token_embeddings, estimated_token_embeddings2, return_last_states=True)

    results = client.query(embedding_input=text_embeddings[0].tolist())

    output = ''

    for idx, result in enumerate(results):
        image_url = result['url']
        output += f'![image]({image_url})\n'


    if do_generate:
        text_embeddings, text_last_hidden_states = encode_with_pseudo_tokens_HF(model_dict['decoder'].text_encoder, input_tokens, estimated_token_embeddings, return_last_states=True)
        images = model_dict['decoder'](num_inference_steps=25,
                                      num_images_per_prompt=1,
                                      prompt_embeds=text_last_hidden_states).images
    else:
        images = []

    return images, output


def test_fps(batch_size=1):
    dummy_images = torch.rand([batch_size, 3, 224, 224])

    todo_list = ['phi', 'phi_pic2word']

    input_tokens = model_dict['tokenizer'](text=['a photo of $1 with flowers'] * batch_size, return_tensors='pt', padding='max_length', truncation=True)['input_ids'].to(model_dict['device'])
    input_tokens = torch.where(input_tokens == 49409,
                               torch.ones_like(input_tokens) * 259,
                               input_tokens)


    for model_name in todo_list:
        time_array = []
        n_repeat = 100
        for _ in range(n_repeat):
            start_time = time.time()
            image_features = model_dict['clip_vision_model'](pixel_values=dummy_images.to(model_dict['clip_vision_model'].device, dtype=model_dict['clip_vision_model'].dtype)).image_embeds
            token_embeddings = model_dict[model_name](image_features)
            text_embeddings = encode_with_pseudo_tokens_HF(model_dict['clip_text_model'], input_tokens, token_embeddings)
            end_time = time.time()
            if _ > 5:
                time_array.append(end_time - start_time)
        print(f"{model_name}: {np.mean(time_array):.4f}")

if __name__ == '__main__':
    args = parse_args()

    global model_dict, client

    model_dict = load_models(args)

    if args.test_fps:
        # check FPS of all models.
        test_fps(1)
        exit()


    client = ClipClient(url="https://knn.laion.ai/knn-service",
                        indice_name="laion5B-H-14" if args.clip_model_name == "huge" else "laion5B-L-14",
                        )

    title = 'Zeroshot CIR demo'

    with gr.Blocks(title=title) as demo:
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    image_source1 = gr.Image(type='pil', label='image1')
                    image_source2 = gr.Image(type='pil', label='image2')
                model_name = gr.Radio(['ours', 'searle', 'pic2word'], label='phi model', value='ours')
                text_input = gr.Textbox(value='', label='Input text guidance. $1, $2')
                submit_button = gr.Button('Submit')
            with gr.Column():
                do_generate = gr.Checkbox(value=False, label='generate an image with SD-2-1', visible=False)
                gallery = gr.Gallery(label='Generated images', visible=False).style(colums=2, object_fit='contain')
                md_output = gr.Markdown(label='Output')

        submit_button.click(predict, inputs=[image_source1, image_source2, text_input, do_generate, model_name], outputs=[gallery, md_output])

    demo.launch(server_name='0.0.0.0',
                server_port=8000)
