# Third-party components and provenance

- Frozen features use [DINOv2](https://github.com/facebookresearch/dinov2) through [timm](https://github.com/huggingface/pytorch-image-models). Pretrained weights are downloaded from their original distributors or supplied locally; no model weights are bundled.
- Hashes use [ImageHash](https://github.com/JohannesBuchner/imagehash); RANSAC and image operations use OpenCV. Their licenses apply to those packages.
- Optional training invokes [Ultralytics](https://github.com/ultralytics/ultralytics), whose upstream package has its own AGPL-3.0/commercial licensing terms. Its source or checkpoints are not bundled in this repository.
- SSCD result records refer to the [official copy-detection model](https://github.com/facebookresearch/sscd-copy-detection), with its preprocessing and normalization variants retained in the released metadata.
- Synthetic corruption settings and the deterministic compound-transform implementation are adapted from the authors' MIT-licensed DeQCal code. Its attribution is reproduced in `licenses/DeQCal-MIT.txt`.
- Benchmark images and annotations remain with their distributors; dataset provenance is in `docs/DATASETS.md`.

The HashDV license applies to the original code in this repository and does not relicense third-party software, models or datasets.
