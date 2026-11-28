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

## Run 3: balanced sampling + label smoothing (12 epochs, ~145 s/epoch)
- Changes vs Run 2: sqrt-inverse-frequency WeightedRandomSampler and label smoothing 0.1.
- Best epoch (7), validation: accuracy 0.860, macro-F1 0.760.
- Recall: CBB 0.570, CBSD 0.763, CGM 0.757, CMD 0.935, Healthy 0.770.
- Diseased leaves called Healthy: 6.6% (Run 2: 5.4%).
- Finding: the rare-disease fix helped CBB (about 6 more leaves of 107) but cost recall
  on CMD and raised the dangerous mistake. The macro-F1 difference (0.008) is inside
  the epoch-to-epoch noise (validation macro-F1 moved between 0.72 and 0.76 within a run).
- Both runs overfit: training accuracy ~0.91 vs validation ~0.86.

## Averaging Run 2 and Run 3 (validation set)
- Accuracy 0.869, macro-F1 0.765, diseased called Healthy 5.8%.
- Gain over the single models is small (about 11 photos of 2,137) and it doubles the
  model size and phone inference time.
- Decision: the app will use ONE model, not an ensemble.

## Evaluation data
- results/eval_data.npz holds both models' probabilities for the validation and
  test sets (saved without printing test scores).
- The test set is sealed. It will be used once, at the end, to report final numbers.

## Decisions and next steps
- Keep EfficientNet-B0 (about 16 MB, suits low-end phones).
- Optional experiments if time remains after the app works: MobileNetV3-Large at 256px,
  EfficientNet-B0 at 384px.
- Next: design a "retake the photo" rule on the validation set (confidence threshold,
  stricter for the Healthy answer), then check it once on the test set.

  
## Guardrail design and calibration (validation set only)
- Temperature scaling fitted on validation: Run 2 T=1.5 (was overconfident),
  Run 3 T=0.8 (underconfident, due to label smoothing), average T=0.9.
- Calibration error (ECE): Run 2 5.1% -> 1.5%, Run 3 6.6% -> 1.7%, average 2.8% -> 1.6%.
- "Healthy only if confidence >= 0.90" rule, calibrated, general threshold 0:
  Run 2: sick leaves called Healthy 5.4% -> 0.3%, photos answered 88.5%, accuracy of answers 89.7%.
  Run 3: 6.6% -> 1.1%. Average: 5.8% -> 0.6%.
- Run 2 had the fewest dangerous mistakes at every Healthy level tested (0.70-0.95).
  The gap is about 15 leaves of ~1,880 and could be partly noise.
- DECISION: the app uses Run 2 alone (EfficientNet-B0, ~16 MB, T=1.5).
  Rule: answer "Healthy" only at >= 0.90 calibrated confidence; any answer below 0.50 asks for a retake.
- Caveat: thresholds were chosen on the validation set. Final numbers will come from
  one evaluation on the sealed test set.

  
## ONNX export (Run 2)
- Run 2 (EfficientNet-B0) exported to ONNX: 16.0 MB, variable batch size.
- PyTorch vs ONNX Runtime: largest score difference 0.000003, same predicted classes.
- CPU speed on my laptop (i7-1255U): median 18 ms per photo, slowest 90 ms.
  Not measured on a phone, so no phone speed is claimed.

## Grad-CAM (14 validation photos, qualitative only)
- Grad-CAM on the last feature maps (8x8 grid stretched to 256x256).
- Confident correct disease predictions mostly highlight discoloured leaf areas.
  Uncertain predictions (about 50% or less) show diffuse or off-leaf attention.
- One photo labelled Healthy shows a clearly yellow leaf (model: CBB 50%): possible label noise or ambiguity.
- Limits: coarse resolution; shows where the model looked, not proof it is right;
  14 hand-picked photos only; the model may also use background cues (not tested).

## App design decisions
- Three answer levels: (1) confident diagnosis with advice; (2) "probably X, not sure,
  retake the photo or ask an extension officer"; (3) "retake the photo" (very low
  confidence or blurry image).
- "Healthy" needs calibrated confidence >= 0.80 by default, adjustable.
  Trade-off on validation (Run 2, calibrated): 0.70 -> 92.9% photos answered, 1.5% sick leaves
  called Healthy; 0.80 -> 91.0%, 0.7%; 0.90 -> 88.5%, 0.3%.
- The final threshold is fixed before the single test-set check.


## Thresholds (validation set, Run 2 calibrated, T=1.5)
- "Not sure" below 0.60 confidence; "Healthy" needs at least 0.80.
- On validation: 82.4% of photos get a confident answer, 92.7% of those are correct
  (about 6 wrong confident answers per 100 photos), and 0.7% of sick leaves get a confident "Healthy".
- Chosen as a trade-off between wrong answers and "not sure" results. Not yet checked on real phone photos
  or on the sealed test set.

## Inference engine and app
- app/engine.py: quality check -> ONNX prediction -> calibration -> decision
  (retake / not sure / cannot confirm healthy / confident), plus Grad-CAM on request.
- app/streamlit_app.py: upload or camera, result, advice, heatmap checkbox, English/Kinyarwanda switch.
- All screen texts and advice are in app/i18n_en.json and app/i18n_rw.json.
- Advice written from published sources (IITA, journal papers, extension factsheets); the sources are listed in
  app/advice.json. No reliable management advice was found for Cassava Green Mottle (CGM), so the app says it is
  not sure of the cause and sends the farmer to an extension officer.
- Kinyarwanda translation by the author (native speaker). Not yet reviewed by an agronomist or extension officer.

## Informal app checks
- Two photos from outside the dataset (one stock photo with a watermark) ran through the app in both languages.
  Results were not evaluated, only checked that the app works.
- Known gap: the model has only 5 classes, so it cannot say "other disease" or "not a cassava leaf".


## Test with photos outside cassava (14 photos, taken or collected by the author)
- Photos: other leaves (avocado, banana, mango, papaya, beans), objects (shoe, ball, chair, desk, hand) and soil, water.
- App result without any "not cassava" check: 4 confident answers (avocado -> Healthy 0.83, banana -> Healthy 0.96,
  soil -> CMD 0.67, soil1 -> CBSD 0.96), 4 "cannot confirm healthy", 4 "not sure", 2 "retake".
- Tried a distance check in the model's 1,280-number feature space (Mahalanobis distance to the nearest class average,
  limit = 99th percentile of validation photos). It rejected only 1 of 14 (water) and 0 of 14 real cassava photos.
  Mango, papaya, beans and avocado were closer to cassava than the median cassava photo.
- Conclusion: features trained only to separate cassava diseases do not separate cassava from other plants.
- Decision: show a visible notice ("This app only checks cassava leaves...") in both languages and report the limitation.
  Possible future work: a dedicated "not cassava" class trained with photos of other plants, or a general-purpose
  pretrained model's features. Neither has been tried.