## LinCIR: Language-only Training of Zero-shot Composed Image Retrieval (CVPR 2024)

[![arXiv](https://img.shields.io/badge/arXiv%20papr-2312.01998-b31b1b.svg)](https://arxiv.org/abs/2312.01998)
[![demo](https://img.shields.io/badge/Demo-Link-blue.svg)](https://huggingface.co/spaces/navervision/LinCIR)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/language-only-efficient-training-of-zero-shot/zero-shot-composed-image-retrieval-zs-cir-on-2)](https://paperswithcode.com/sota/zero-shot-composed-image-retrieval-zs-cir-on-2?p=language-only-efficient-training-of-zero-shot) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/language-only-efficient-training-of-zero-shot/zero-shot-composed-image-retrieval-zs-cir-on)](https://paperswithcode.com/sota/zero-shot-composed-image-retrieval-zs-cir-on?p=language-only-efficient-training-of-zero-shot) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/language-only-efficient-training-of-zero-shot/zero-shot-composed-image-retrieval-zs-cir-on-1)](https://paperswithcode.com/sota/zero-shot-composed-image-retrieval-zs-cir-on-1?p=language-only-efficient-training-of-zero-shot)

By combining LinCIR with [RTD](https://arxiv.org/abs/2406.09188), we can achieve:

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/reducing-task-discrepancy-of-text-encoders/zero-shot-composed-image-retrieval-zs-cir-on-2)](https://paperswithcode.com/sota/zero-shot-composed-image-retrieval-zs-cir-on-2?p=reducing-task-discrepancy-of-text-encoders) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/reducing-task-discrepancy-of-text-encoders/zero-shot-composed-image-retrieval-zs-cir-on)](https://paperswithcode.com/sota/zero-shot-composed-image-retrieval-zs-cir-on?p=reducing-task-discrepancy-of-text-encoders) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/reducing-task-discrepancy-of-text-encoders/zero-shot-composed-image-retrieval-zs-cir-on-1)](https://paperswithcode.com/sota/zero-shot-composed-image-retrieval-zs-cir-on-1?p=reducing-task-discrepancy-of-text-encoders)


Welcome to the official Pytorch implementation of LinCIR!

Discover the magic of **LinCIR**, a ground-breaking approach to Composed Image Retrieval (CIR) that challenges convention and ushers in a new era of AI research. Dive into the limitless possibilities of zero-shot composed image retrieval with us!

**Authors**: 

**[Geonmo Gu](https://geonm.github.io/)\*<sup>1</sup>, [Sanghyuk Chun](https://sanghyukchun.github.io/home/)\*<sup>2</sup>, [Wonjae Kim](https://wonjae.kim)<sup>2</sup>, [Yoohoon Kang](https://www.linkedin.com/in/yoohoon-kang-59895644)<sup>1</sup>, [Sangdoo Yun](https://sangdooyun.github.io)<sup>2</sup>**

<sup>1</sup> NAVER Vision <sup>2</sup> NAVER AI Lab

\* First two authors contributed equally.

## ⭐ Overview
<img src=".github/figure_001.png" height="256">

The Composed Image Retrieval (CIR) task, a fusion of image and text, has always been an intriguing challenge for AI researchers. Traditional CIR methods require expensive triplets of query image, query text, and target image for training, limiting scalability.

Enter LinCIR, a revolutionary CIR framework that relies solely on language for training. Our innovative approach leverages self-supervision through self-masking projection (SMP), allowing LinCIR to be trained using text datasets alone.

With LinCIR, we achieve astonishing efficiency and effectiveness. For instance, LinCIR with a CLIP ViT-G backbone is trained in just 48 minutes and outperforms existing methods in zero-shot composed image retrieval on four benchmark datasets: CIRCO, GeneCIS, FashionIQ, and CIRR. In fact, it even surpasses supervised methods on FashionIQ!

## 🚀 News
- **February 27, 2024** - LinCIR is accepted at CVPR 2024!
- **December 5, 2023** - LinCIR is officially released!

## 🛠️ Installation
Get started with LinCIR by installing the necessary dependencies:

```bash
$ pip install torch transformers diffusers accelerate datasets spacy
$ python -m spacy download en_core_web_sm
```

## 🤗 Demo
If you want to run and execute the demo directly, you can do so by running the script below.

Of course, you can also experience the demo directly on the [Huggingface Space](https://huggingface.co/spaces/navervision/LinCIR).

```bash
$ pip intall clip-retrieval

$ python demo.py
```

Demo will be hosted at https://0.0.0.0:8000

## 📂 Dataset Preparation
No need to worry about downloading training datasets manually. All training datasets are automatically fetched using the Hugging Face datasets library.

Keep in mind that the training datasets are considerably smaller in volume compared to (image, caption) pairs or triplet datasets like FashionIQ and CIRR. 

Please refer to [here](https://github.com/miccunifi/SEARLE/tree/main#data-preparation) to prepare the benchmark datasets.

## 📚 How to Train LinCIR
Train LinCIR with ease using the following command:

```bash
$ python -m torch.distributed.run --nproc_per_node 8 --nnodes 1 --node_rank 0 \
--master_addr localhost --master_port 5100 train_phi.py \
--batch_size 64 \
--output_dir /path/to/your_experiment \
--cirr_dataset_path /path/to/cir_datasets/CIRR \
--mixed_precision fp16 \
--clip_model_name large \
--validation_steps 1000 \
--checkpointing_steps 1000 \
--seed 12345 \
--lr_scheduler constant_with_warmup --lr_warmup_steps 0 \
--max_train_steps 20000
```

If you have a powerful GPU machine with 8 GPUs, simply run the above script. For less powerful GPU machine with single GPU, set `--nuproc_per_node` to 1 and adjust `--batch_size` to 256 or 512. Rest assured, the results will be consistent.

If you'd like to use ViT-Large, Huge or Giga as CLIP backbone, change `--clip_model_name` to large, huge, or giga each.

## 💯 How to Evaluate LinCIR

### CIRR (Test Set)
Evaluate LinCIR on the CIRR test set with the following command:

```bash
$ python generate_test_submission.py \
--eval-type phi \
--dataset cirr \
--dataset-path /path/to/CIRR \
--phi-checkpoint-name /path/to/trained_your/phi_best.pt \
--clip_model_name large \
--submission-name lincir_results
```

Retrieved results will be saved as:
- `./submission/cirr/{submission-name}.json`
- `./submission/cirr/subset_{submission-name}.json`

Upload these files [here](https://cirr.cecs.anu.edu.au/test_process/) to view the results.

### CIRR (Validation Set, Dev)
For the CIRR validation set, use the following command:

```bash
$ python validate.py \
--eval-type phi \
--dataset cirr \
--dataset-path /path/to/CIRR \
--phi-checkpoint-name /path/to/trained_your/phi_best.pt \
--clip_model_name large
```

### FashionIQ
To evaluate LinCIR on FashionIQ, run the following command:

```bash
$ python validate.py \
--eval-type phi \
--dataset fashioniq \
--dataset-path /path/to/fashioniq \
--phi-checkpoint-name /path/to/trained_your/phi_best.pt \
--clip_model_name large
```

### CIRCO
Evaluate LinCIR on the CIRCO dataset with the command below:

```bash
$ python generate_test_submission.py \
--eval-type phi \
--dataset circo \
--dataset-path /path/to/cir_datasets/CIRCO \
--phi-checkpoint-name /path/to/trained_your/phi_best.pt \
--clip_model_name large \
--submission-name lincir_results
```

Retrieved results will be saved as:
- `./submission/circo/{submission-name}.json`
- `./submission/circo/subset_{submission-name}.json`

Upload these files [here](https://circo.micc.unifi.it/evaluation) to view the results.

### GeneCIS
Evaluating GeneCIS requires a few additional steps. Run the following script:

You can get `VG_100K_all` and `COCO_val2017` at [GeneCIS](https://github.com/facebookresearch/genecis?tab=readme-ov-file#-arrow_down-downloads).

```bash
# Assuming you're in the lincir folder.
$ git fetch --all
$ git checkout eval_genecis
$ cd genecis
$ python evaluate.py \
--combiner_mode phi \
--model large \
--combiner_pretrain_path /path/to/lincir_best.pt \
--vg_100k_all_path /path/to/VG_100K_all \
--coco_val2017_path /path/to/val2017
```

## Acknowledgement
We would like to express our special gratitude to the authors of [SEARLE](https://github.com/miccunifi/SEARLE) for their invaluable contributions, as our code draws significant inspiration from this open-source project.

## Citation
```
@inproceedings{gu2024lincir,
    title={Language-only Training of Zero-shot Composed Image Retrieval},
    author={Gu, Geonmo and Chun, Sanghyuk and Kim, Wonjae and and Kang, Yoohoon and Yun, Sangdoo},
    year={2024},
    booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
}
```

## License

Licensed under [CC BY-NC 4.0](LICENSE)

```
LinCIR
Copyright (c) 2023-present NAVER Corp.
CC BY-NC-4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
```
