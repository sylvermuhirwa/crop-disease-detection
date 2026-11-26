# Project Log: Cassava Disease Triage Assistant

## Setup
- Local: Windows 11, Python 3.13, VS Code, virtual environment (.venv).
- Training: Kaggle notebooks, free Tesla T4 GPU.
- Kaggle sessions were unstable on my first network (stuck editor, kernel crashes).
  Run 2 completed after I switched networks.

## Dataset
- Makerere AI Lab / NaCRRI Cassava Leaf Disease Classification (Kaggle), Uganda field photos.
- 21,397 images (all 800x600, 0 missing, 0 corrupt), 5 classes:
  CBB 1,087 (5.1%), CBSD 2,189, CGM 2,386 (Cassava Green Mottle), CMD 13,158 (61.5%), Healthy 2,577.
- Class names are read from the dataset's own label_num_to_disease_map.json.
- Decision: judge models by macro-F1 and per-class recall, not accuracy alone.
- Observation: some labels look doubtful (e.g. a "Healthy" photo with brown, damaged leaves).

## Splits and duplicates
- Stratified 80/10/10, random_state=42: train 17,117 / val 2,140 / test 2,140.
- Perceptual hash (dHash, 64 bits, Hamming distance <= 6) found 3 near-duplicate pairs
  crossing splits. One pair had conflicting labels (evidence of label noise).
- Fix: kept the training copy, removed the others. Final: train 17,117 / val 2,137 / test 2,140.
- Limitation: only near-identical copies are detected, not different photos of the same plant.

## Data pipeline
- PyTorch + timm, image size 256x256, batch size 32, ImageNet normalization.
- Train-only augmentation: random crop (scale 0.6-1.0), flips, rotation 20 deg,
  brightness/contrast/saturation jitter, blur. No hue shift (leaf colour is diagnostic).
- Validation and test images are only resized.

## Baseline (5 epochs)
- EfficientNet-B0, ImageNet pretrained, 4.01M parameters, AdamW lr 3e-4, OneCycle,
  mixed precision, plain cross-entropy.
- Best epoch (4), validation: accuracy 0.841, macro-F1 0.721.
- Recall: CBB 0.505, CBSD 0.648, CGM 0.682, CMD 0.943, Healthy 0.774.
- 152 of 1,880 diseased validation leaves (8.1%) were predicted Healthy.

## Run 2: longer training (12 epochs, ~140 s/epoch)
- Best epoch (11), validation: accuracy 0.864, macro-F1 0.752.
- Recall: CBB 0.514, CBSD 0.749, CGM 0.741, CMD 0.957, Healthy 0.743.
- Diseased leaves called Healthy: 8.1% -> 5.4%.
- Overfitting starts after ~epoch 8 (validation loss stops falling).
- CBB recall barely changed (about one image). Validation scores are slightly
  optimistic because the best epoch was chosen on them.

## Limitations so far
- Label noise, single-country data (Uganda), no severity grading, no field trial.