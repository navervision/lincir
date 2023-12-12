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
from huggingface_hub import hf_hub_url, cached_download


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--lincir_ckpt_path", default=None, type=str,
                        help="The output directory where the model predictions and checkpoints will be written")
    parser.add_argument("--pic2word_ckpt_path", default=None, type=str)
    parser.add_argument("--cache_dir", default="./hf_models", type=str,
                        help="Path to model cache folder")
    parser.add_argument("--clip_model_name", default="large", type=str,
                        help="CLIP model to use, e.g 'large', 'huge', 'giga'")
    parser.add_argument("--mixed_precision", default="fp16", type=str)
    parser.add_argument("--test_fps", action="store_true")
    args = parser.parse_args()
    return args


def load_models(args):
    if torch.cuda.is_available():
        device = 'cuda:0'
        dtype = torch.float16
    else:
        device = 'cpu'
        dtype = torch.float32

    clip_vision_model, clip_preprocess, clip_text_model, tokenizer = build_text_encoder(args)

    tokenizer.add_special_tokens({'additional_special_tokens':["[$]"]}) # 49408

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

    clip_vision_model.to(device, dtype=dtype)
    clip_text_model.to(device, dtype=dtype)

    if not args.test_fps:
        # download and load sd
        if not os.path.exists('./pretrained_models/lincir_large.pt'):
            model_file_url = hf_hub_url(repo_id='navervision/zeroshot-cir-models', filename='lincir_large.pt')
            cached_download(model_file_url, cache_dir='./pretrained_models', force_filename='lincir_large.pt')
        state_dict = torch.load('./pretrained_models/lincir_large.pt', map_location=device)
        phi.load_state_dict(state_dict['Phi'])

        if not os.path.exists('./pretrained_models/pic2word_large.pt'):
            model_file_url = hf_hub_url(repo_id='navervision/zeroshot-cir-models', filename='pic2word_large.pt')
            cached_download(model_file_url, cache_dir='./pretrained_models', force_filename='pic2word_large.pt')
        sd = torch.load('./pretrained_models/pic2word_large.pt', map_location=device)['state_dict_img2text']
        sd = {k[len('module.'):]: v for k, v in sd.items()}
        phi_pic2word.load_state_dict(sd)

    phi.to(device, dtype=dtype)
    phi_searle.to(device, dtype=dtype)
    phi_pic2word.to(device, dtype=dtype)

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
            'clip_model_name': args.clip_model_name,
            }


def predict(images, input_text, model_name):
    start_time = time.time()
    input_images = model_dict['clip_preprocess'](images, return_tensors='pt')['pixel_values'].to(model_dict['device'])
    input_text = input_text.replace('$', '[$]')
    input_tokens = model_dict['tokenizer'](text=input_text, return_tensors='pt', padding='max_length', truncation=True)['input_ids'].to(model_dict['device'])
    input_tokens = torch.where(input_tokens == 49408,
                               torch.ones_like(input_tokens) * 259,
                               input_tokens)
    image_features = model_dict['clip_vision_model'](pixel_values=input_images.to(model_dict['dtype'])).image_embeds
    clip_image_time = time.time() - start_time

    start_time = time.time()
    if model_name == 'lincir':
        estimated_token_embeddings = model_dict['phi'](image_features)
    elif model_name == 'searle':
        estimated_token_embeddings = model_dict['phi_searle'](image_features)
    else: # model_name == 'pic2word'
        estimated_token_embeddings = model_dict['phi_pic2word'](image_features)
    phi_time = time.time() - start_time

    start_time = time.time()
    text_embeddings, text_last_hidden_states = encode_with_pseudo_tokens_HF(model_dict['clip_text_model'], input_tokens, estimated_token_embeddings, return_last_states=True)
    clip_text_time = time.time() - start_time

    start_time = time.time()
    results = client.query(embedding_input=text_embeddings[0].tolist())
    retrieval_time = time.time() - start_time

    output = ''

    for idx, result in enumerate(results):
        image_url = result['url']
        output += f'![image]({image_url})\n'

    time_output = {'CLIP visual extractor': clip_image_time,
                   'CLIP textual extractor': clip_text_time,
                   'Phi projection': phi_time,
                   'CLIP retrieval': retrieval_time,
                   }
    setup_output = {'device': model_dict['device'],
                    'dtype': model_dict['dtype'],
                    'Phi': model_name,
                    'CLIP': model_dict['clip_model_name'],
                    }

    return {'time': time_output, 'setup': setup_output}, output


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

    md_title = f'''# {title}
    [LinCIR](https://arxiv.org/abs/2312.01998): Language-only Training of Zero-shot Composed Image Retrieval  
    [SEARLE](https://arxiv.org/abs/2303.15247): Zero-shot Composed Image Retrieval with Textual Inversion  
    [Pic2Word](https://arxiv.org/abs/2302.03084): Mapping Pictures to Words for Zero-shot Composed Image Retrieval  

    K-NN index for the retrieval results are entirely trained using the entire Laion-5B imageset. This is made possible thanks to the great work of [rom1504](https://github.com/rom1504/clip-retrieval).
    '''

    with gr.Blocks(title=title) as demo:
        gr.Markdown(md_title)
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    image_source = gr.Image(type='pil', label='image1')
                model_name = gr.Radio(['lincir', 'searle', 'pic2word'], label='Phi model', value='lincir')
                text_input = gr.Textbox(value='', label='Input text guidance. Special token is $')
                submit_button = gr.Button('Submit')
                gr.Examples([["example1.jpg", "$, pencil sketch", 'lincir']], inputs=[image_source, text_input, model_name])
            with gr.Column():
                json_output = gr.JSON(label='Processing time')
                md_output = gr.Markdown(label='Output')

        submit_button.click(predict, inputs=[image_source, text_input, model_name], outputs=[json_output, md_output])

    demo.queue()

    demo.launch(server_name='0.0.0.0',
                server_port=8000)
