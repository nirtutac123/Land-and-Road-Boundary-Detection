# Lane and Road Boundary Detection - Q&A and Presentation Guide

Use this as a speaking guide, not as a script to memorise word-for-word. Every answer below matches the final notebook: `Lane_and_Road_Boundary_Detection_Final.ipynb`.

## 30-second project introduction

Our project is **Lane and Road Boundary Detection under Weather and Occlusion**. The aim is to identify lane pixels in road images even when the scene is difficult because of glare, shadows, vehicles, worn markings, or low light. We used the CULane dataset. We compared two simple, training-free computer-vision baselines with one genuinely trained machine-learning pixel classifier. We used the official train, validation, and test lists, evaluated with pixel-level precision, recall, F1, IoU, and accuracy, then inspected results by test scenario. The main message is that this is a lightweight and reproducible baseline study; it is not a production lane-detection system and it does not claim to be a CNN.

---

## A. Title and problem understanding

### Q1. What is the title of the project?

**Lane and Road Boundary Detection under Weather and Occlusion.**

### Q2. What problem are you solving?

We are solving a pixel-level segmentation problem. For each road image, the output is a binary mask: lane pixel or background pixel. We compare different ways of producing that mask.

### Q3. Why is lane detection important?

Lane information is useful for driver-assistance and autonomous-driving systems. However, a road image is difficult in practice: lane markings can be hidden by vehicles, faded, affected by glare or shadows, curved, or absent.

### Q4. What does “under weather and occlusion” mean in your project?

It means we do not only look at easy road scenes. CULane includes difficult categories such as crowd/vehicle occlusion, highlight/glare, shadow, no-line, arrow, curve, cross-road, and night. We use the scenario lists to inspect how methods behave under different visual conditions.

### Q5. Is this classification, detection, or segmentation?

It is **binary semantic segmentation at pixel level**. The model predicts whether every pixel belongs to a lane marking. It is not object detection with bounding boxes.

---

## B. Dataset and EDA

### Q6. Which dataset did you use?

CULane, a road-lane dataset. It contains driving images, lane annotations, official split lists, and segmentation masks.

### Q7. What files did you use from CULane?

- Road images from the driver folders.
- Official list files: `train.txt`, `val.txt`, `test.txt`, and category files inside `list/test_split`.
- Segmentation masks from `laneseg_label_w16` and `laneseg_label_w16_test`.
- Line annotation files in `annotations_new` for visual understanding of lanes.

### Q8. What is the label/target?

The segmentation-mask value is converted to a binary target. Any non-zero mask value is treated as a lane pixel; zero is background. This gives one binary target per image pixel.

### Q9. What did you do in EDA?

The notebook reads the official split lists, counts official entries and locally available labelled images, and displays validation images beside their ground-truth masks. This verifies that images and labels align before modelling.

### Q10. Why did you manually visualise images and masks first?

Counts alone are not enough. By viewing image-mask pairs and overlays, we can catch path mistakes, wrong labels, missing files, or masks that do not align with lanes. This is especially important in a computer-vision project.

### Q11. Why does the notebook mention missing local entries?

The full CULane data is very large, and the local copy does not contain every image referenced by the official lists. The notebook filters only entries whose image and mask actually exist locally. It reports the counts honestly instead of silently failing or inventing data.

### Q12. What are the risks in this dataset?

Lane pixels are much rarer than background pixels, nearby video frames are correlated, and difficult scenes may have low visibility or unclear lane markings. Therefore, accuracy alone can be misleading and random splitting can create leakage.

---

## C. Methodology and split design

### Q13. Explain your workflow from beginning to end.

1. Locate the dataset with relative paths and fix a random seed.
2. Read the official train, validation, and test lists.
3. Check available images and masks and visually inspect samples.
4. Convert masks to binary lane/background targets.
5. Build two classical baselines and one supervised ML model.
6. Train the ML model on a deterministic subset of training images.
7. Compare all methods on the validation subset using the same metrics.
8. Inspect labelled test scenarios separately for error analysis.
9. Visualise predictions and report limitations.

### Q14. What do your handwritten V1 and V2 split notes mean?

**V1** is effectively train versus test only. It is weak because there is no independent validation data for choosing a method or settings. If we repeatedly check the test set while deciding, we overfit our decisions to the test set.

**V2** is train / validation / test. This is the methodology we use. We train on training data, compare/select methods using validation data, and reserve test scenarios for final error analysis. This is the correct experimental design for a fair comparison.

### Q15. Why did you not randomly split individual frames?

Consecutive frames from driving videos are visually similar. A random frame split can put near-duplicate scenes in both training and evaluation, which makes performance look better than it really is. We use the official CULane lists to respect the dataset's intended split structure.

### Q16. Why did you use a subset instead of all 88,000+ training images?

The dataset is large and the presentation must run smoothly. We use a reproducible, evenly spaced subset of 32 locally available training images. The notebook states this limitation clearly. The goal is a real, demonstrable baseline experiment, not a claim of full-scale training.

### Q17. How is your subset reproducible?

The random seed is fixed to 42. Images are selected deterministically using evenly spaced positions from the available training entries. Pixel sampling also uses the same seed.

### Q18. How did you handle class imbalance?

Most pixels are background. For each selected image, we sample lane pixels and background pixels in a balanced way when constructing the supervised training set. Without this, a model could predict mostly background and still obtain high accuracy.

---

## D. Models

### Q19. Which three models did you compare?

1. **Colour-threshold baseline:** HSV rules for bright white/yellow lane-like pixels.
2. **Canny + Hough baseline:** edge detection followed by Hough line segments.
3. **Histogram Gradient Boosting pixel classifier:** a supervised scikit-learn model trained from RGB and position features.

### Q20. Why did you include simple baselines?

A simple correct baseline tells us whether the trained method adds value. Classical methods are fast, interpretable, and appropriate for a first computer-vision comparison. A more complex method is only meaningful if it is compared fairly with a baseline.

### Q21. Is your trained model a CNN?

**No.** It is a `HistGradientBoostingClassifier` from scikit-learn. It is an honest lightweight pixel classifier, not a convolutional neural network. It learns from RGB colour and normalized x/y pixel coordinates.

### Q22. Why did you use Histogram Gradient Boosting?

It is a genuine supervised model that trains quickly on tabular pixel features, works on a laptop, and is easy to save and reproduce. It was suitable for our constrained live demonstration. Its limitation is that it does not use the spatial neighbourhood as effectively as a real CNN or U-Net.

### Q23. What features does the trained model use?

For each pixel: normalized red, green, and blue values, plus normalized horizontal position `x` and vertical position `y`. Position helps because lanes are more likely in particular regions of a road image.

### Q24. How do the models make predictions?

The colour baseline applies fixed HSV thresholds. The Hough baseline finds image edges and draws detected line segments. The trained classifier predicts lane/background for each pixel based on the learned feature patterns.

### Q25. Did you save a trained model?

Yes. The notebook saves the fitted model as `outputs/culane_pixel_classifier.joblib`. It also writes `outputs/training_run_log.csv` with start time, end time, elapsed time, seed, images used, and sampled pixels.

---

## E. Training, metrics, and results

### Q26. Was anything simulated or randomly generated in the final notebook?

No. The final notebook uses a fixed seed only for reproducible sampling. The model is fitted using `model.fit(...)`; metrics are calculated from predicted masks and ground-truth masks. There are no random losses, random metrics, or fake CNN claims.

### Q27. How long did training take?

The most recent verified run trained on 32 images and 105,000 sampled pixels. The fitted model took **about 0.65 seconds** on this laptop environment. The exact start time, end time, and elapsed seconds are logged by the notebook; they should be quoted from the displayed run log during presentation.

### Q28. Which metrics did you use, and why?

- **Precision:** whether predicted lane pixels are really lanes.
- **Recall:** whether true lane pixels were found.
- **F1-score:** balances precision and recall; useful because lane pixels are sparse.
- **IoU:** measures overlap between the predicted and true lane regions.
- **Accuracy:** included for completeness, but not treated as the main metric because background dominates.

### Q29. Why is F1 more useful than accuracy here?

If 95% of pixels are background, a model can predict background everywhere and still appear accurate. F1 focuses on the lane class by balancing missed lanes and false lane predictions.

### Q30. What was the main validation result?

In the most recent verified deterministic run, the **trained pixel classifier** had the best validation F1, approximately **0.1616**. This is a modest score, and we report it honestly. It demonstrates that the trained model improved on the tested simple baselines in this limited experiment, not that it is ready for deployment.

### Q31. Why is the F1-score modest?

Lane segmentation is difficult, the model is deliberately lightweight, it trains on only 32 images, and it uses only independent pixel colour/position features. It does not learn detailed spatial lane structure, temporal information, or robust high-level features like a CNN would.

### Q32. How did you use the test data?

We did not use test data to choose the model. We used validation data for comparison/model selection. The official test-category lists are used as a small labelled scenario analysis to understand behaviour under normal, crowd, glare, shadow, no-line, arrow, curve, cross-road, and night conditions.

### Q33. Why do you only use one labelled image per test scenario in the live notebook?

It keeps the demonstration responsive. It is a qualitative/error-analysis sample, not a claim of final benchmark performance. The notebook clearly says validation is the 20-image comparison set and scenario testing is a small deterministic analysis.

---

## F. Visualisation, error analysis, and limitations

### Q34. What did you visualise?

The notebook visualises image-mask pairs during EDA and then shows the original image, ground truth, colour-threshold prediction, Hough prediction, and trained-classifier prediction side by side. This makes failures visible rather than relying only on a table.

### Q35. What kinds of failure do you expect?

- Glare, highlights, and shadows can create false edges or change lane colour.
- Vehicles can hide lane markings, causing false negatives.
- Worn, curved, or missing markings are hard for both fixed rules and simple pixel features.
- Road boundaries, arrows, barriers, and bright objects can create false positives.

### Q36. What are the main limitations of your work?

- It uses a small training subset.
- The trained model treats pixels independently; it has no strong spatial or temporal reasoning.
- It uses pixel metrics, not the official CULane lane-instance evaluator.
- The local data copy is incomplete, so evaluation filters unavailable entries.
- Scenario analysis is intentionally small for a live notebook demonstration.

### Q37. How would you improve the project with more time/resources?

Train a real segmentation network such as U-Net/SCNN on a larger subset or full dataset, use image augmentation for glare/shadow/weather, use the official CULane evaluator, use temporal information across video frames, tune thresholds on validation data only, and analyse each scenario with more samples.

---

## G. Conclusion and presentation checklist

### Q38. What is your conclusion?

The trained lightweight pixel classifier performed best by validation F1 in our reproducible limited experiment, but the result is still modest. Classical methods remain useful as fast baselines. Difficult conditions show why a stronger spatial model and larger-scale evaluation are needed.

### Q39. What is the most important methodological lesson?

Do not trust a single accuracy number or a test-set result used during model selection. First inspect data, use an appropriate train/validation/test design, compare against baselines, calculate real metrics, visualise failures, and state limitations honestly.

### Q40. What should you do just before presenting?

1. Add group number, names, and matriculation numbers in the notebook title cell.
2. Open `Lane_and_Road_Boundary_Detection_Final.ipynb` from the project folder.
3. Confirm `data/CULane` is present locally.
4. Restart the kernel and run all cells once.
5. Verify the model and timing log are created in `outputs/`.
6. Be ready to explain the split, the three models, why F1 matters, the modest result, and limitations.

## Questions you should ask each other

- Can each teammate explain why V2 (train/validation/test) is better than V1 (train/test only)?
- Can each teammate say why this final model is not a CNN?
- Can each teammate explain why F1 and IoU matter more than accuracy for sparse lane pixels?
- Can each teammate explain why the score is modest without pretending it is high?
- Can each teammate point to the real training log and saved `.joblib` model?
- Can each teammate describe one visible failure case and one realistic improvement?
