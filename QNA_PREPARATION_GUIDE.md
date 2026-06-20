# Lane Detection Project: Q&A Preparation Guide

Project: Lane and Road Boundary Detection Under Weather and Occlusion  
Course: Machine Learning, Summer 2026  
Team: Niruta Chapagain, Umesh Singh, Ismail Demircan  
Main demo file: `Lane_Detection_Project.ipynb`

## 1. Thirty-Second Project Explanation

Our project is about detecting lane and road boundaries from dashcam road images. This is useful for driver-assistance and autonomous-driving systems, but it becomes difficult when road conditions are not clean, for example at night, under shadows, glare, occlusion by vehicles, curved roads, intersections, or faded lane markings.

We used the CULane dataset, which contains road images and lane annotations. We followed a four-step methodology: first we visualized the dataset, then performed exploratory data analysis, then used a careful train/validation/test split to avoid leakage, and finally compared three lane-detection approaches: a PolyFit baseline, a CNN-style segmentation approach, and a classical Canny plus Hough line detector.

The main point of the project is not only to get a number, but to understand which method works, where it fails, and why lane detection is challenging in real road scenes.

## 2. One-Minute Presentation Script

The project title is "Lane and Road Boundary Detection Under Weather and Occlusion." We chose this because lane detection is a real-world computer vision problem where simple cases are easy, but difficult scenes are much harder. A lane detector should work not only in normal daylight, but also with glare, shadows, night scenes, curves, crowded traffic, intersections, and missing lane markings.

The dataset is CULane. It contains dashcam road images from different drivers and provides lane annotations as coordinate points. The task is to automatically detect or draw the road lane boundaries.

Our methodology had four main steps. First, we created visualization tools to inspect the images and annotation overlays, because we wanted to understand the data before modelling. Second, we performed EDA: number of images, train/validation/test sizes, annotation structure, and scenario categories. Third, we used the official CULane split instead of a random split, because random image-level splitting can leak similar video frames into both training and testing. Fourth, we compared three models: PolyFit as an interpretable baseline, CNN-style segmentation as a learning-based approach, and Canny plus Hough as a classical computer vision method.

For evaluation, we converted predictions and ground-truth lanes into binary masks and compared them using accuracy, precision, recall, F1-score, and IoU-style overlap. We also did scenario-level error analysis. Our main conclusion is that lane detection is highly scenario-dependent. Simple baselines are useful and explainable, classical methods are fast, and segmentation is the better direction for a stronger real-world system, but it requires proper training and validation.

## 3. Methodology We Followed

This section matches the handwritten methodology notes.

### Step 1: Visualize the Data

What we did:
- Built/used an HTML visualization tool for annotation overlays.
- Displayed road images with lane annotations drawn on top.
- Looked at training, validation, and test examples.
- Checked difficult cases such as night, glare, crowd/occlusion, no-line, curve, crossroad, shadow, and arrow markings.

Why this matters:
- Before training a model, we must know what the dataset actually contains.
- Visualization helps catch annotation problems, missing labels, hard scenes, and unrealistic assumptions.
- It also helps during Q&A because we can show the professor that we manually inspected the data.

How to explain:
"We did not start by training immediately. First, we inspected the images and overlaid lane annotations to understand what the model is supposed to learn. This helped us see why the task is difficult under occlusion, shadows, night driving, and missing lane markings."

### Step 2: Exploratory Data Analysis

What we checked:
- Number of images.
- Number of train, validation, and test samples.
- Test scenario categories.
- Annotation format.
- Driver/video split.
- Whether the dataset has difficult real-world conditions.

Important numbers from the notebook:
- Training samples: 88,880
- Validation samples: 9,675
- Test samples: 34,680
- Total images: 133,235
- Test scenarios: 9

Test categories:
- Normal
- Crowd/occlusion
- Highlight/glare
- Shadow
- No-line
- Arrow
- Curve
- Crossroad
- Night

How to explain:
"EDA showed us that the dataset is not only normal road images. It has separate categories for difficult scenarios. That is useful because we can analyze not only average performance, but also which cases are hardest."

### Step 3: Train/Validation/Test Split

What we considered:
- A random 80/20 split is simple, but risky for video data.
- Road datasets contain many similar frames from the same video.
- If similar frames appear in both train and test, the model may look better than it really is.

What we used:
- Official CULane split.
- Training set, validation set, and scenario-specific test set.
- Test drivers are separate from training drivers.

Why this is better:
- Avoids data leakage.
- Measures generalization to different videos/drivers.
- Matches the professor's requirement to avoid random splits when subject/video/driver leakage is possible.

How to answer if asked about your handwritten "Option 1 vs Option 2":
"We considered splitting by chunks/videos rather than randomly mixing frames. The important idea is that frames from the same video should not be split carelessly across train and test. In the final notebook, we used the official CULane split because it is the safest and most accepted option for this dataset."

### Step 4: Train/Run Models and Compare

Models:
- PolyFit baseline
- CNN-style segmentation
- Classical Canny edge plus Hough line detector

Why three models:
- The professor asked for meaningful baselines and comparison.
- The three models represent different thinking:
  - Geometry-based baseline
  - Learning-based segmentation
  - Classical computer vision

How to explain:
"We did not rely on one method. We compared three approaches so we could discuss trade-offs: interpretability, speed, flexibility, and robustness under difficult conditions."

### Step 5: Visualize Results and Errors

What we showed:
- Model comparison table.
- Metric bar charts.
- Ground-truth lane overlays.
- Predicted lanes/segmentation/lines.
- Scenario difficulty analysis.

Why:
- Lane detection is spatial, so only a table is not enough.
- Visual output helps us see whether predicted lines are actually in the right place.
- Error analysis shows honesty about limitations.

## 4. Dataset Q&A

### Q1. What dataset did you use?

We used the CULane dataset, a lane-detection dataset based on dashcam road images. It includes images and lane annotations, and it has official train, validation, and test splits.

### Q2. What is the target variable?

The target is the lane boundary location. In the raw annotations, lanes are stored as coordinate points. For evaluation, we convert the lane annotations into binary masks so that predicted lane pixels can be compared with ground-truth lane pixels.

### Q3. What does one annotation contain?

One annotation file contains lane point coordinates. Each lane is represented by a sequence of x, y points. These points describe where the lane boundary is in the image.

### Q4. Why is this dataset difficult?

Because it includes real driving conditions. Lane markings may be blocked by cars, hidden by shadows, distorted by glare, missing, curved, or hard to see at night. Intersections and crossroad scenes are also difficult because lane geometry is less clear.

### Q5. How many images are in the dataset?

In our notebook, the dataset summary is:
- 88,880 training samples
- 9,675 validation samples
- 34,680 test samples
- 133,235 images total

### Q6. What are the test scenarios?

The nine CULane test scenarios are normal, crowd, highlight/glare, shadow, no-line, arrow, curve, crossroad, and night.

### Q7. Why did you not use a random split?

Because the dataset is video-based. Adjacent frames can be very similar. If we randomly split images, similar frames from the same video could appear in both training and testing, causing data leakage. We used the official split to make evaluation more realistic.

### Q8. What is data leakage in this project?

Data leakage would happen if the model sees almost the same road scene or video sequence during training and testing. Then the model may memorize scene-specific details instead of learning general lane detection.

### Q9. Why does driver/video separation matter?

It makes testing more realistic. A good model should work on road scenes and drivers it has not already seen.

### Q10. What file types are used?

The dataset mainly uses image files for road scenes and text annotation files for lane coordinates. The notebook reads split files from `data/CULane/list/` and annotation files from `data/CULane/annotations_new/`.

## 5. Model Q&A

### Q11. What models did you compare?

We compared three approaches:
- PolyFit baseline
- CNN-style segmentation
- Classical edge detection plus Hough lines

### Q12. Why did you include PolyFit?

PolyFit is a simple and interpretable baseline. It fits polynomial curves to lane points. It is useful because lane markings often follow smooth curves, but it has limitations when lanes are occluded, missing, or complex.

### Q13. How does PolyFit work?

It takes lane coordinate points and fits a polynomial curve, usually cubic. The curve approximates the lane boundary. In simple road scenes, this can work well because lane lines are often smooth.

### Q14. What is the weakness of PolyFit?

It assumes the lane can be represented as a smooth mathematical curve. It struggles if the lane is partially hidden, broken, missing, or if the road geometry is complex.

### Q15. What is the CNN-style segmentation approach?

It is a learning-based idea where the model predicts lane pixels as a segmentation mask. Instead of fitting only a curve, it tries to identify lane-like pixels in the image.

### Q16. Is the CNN a full production deep-learning model?

No. In the notebook, it is a reduced executable demonstration. It shows the segmentation workflow, but a production-level CNN would need more complete training, hyperparameter tuning, stronger architecture, and longer evaluation.

Safe answer:
"Our notebook demonstrates the segmentation workflow. We are honest that a stronger production CNN would require proper training and tuning beyond this lightweight project demo."

### Q17. Why include classical Canny plus Hough?

It is a fast traditional computer vision baseline. Canny finds edges, and Hough transform finds line segments. It requires no training and helps compare machine learning methods against a simple rule-based approach.

### Q18. How does Canny edge detection work?

Canny detects strong intensity changes in the image. Lane markings often create edges because their color contrasts with the road surface.

### Q19. How does Hough transform help?

After edges are found, Hough transform detects straight line segments among those edge pixels. This is useful for lane lines, especially when roads are mostly straight.

### Q20. What are the weaknesses of Canny plus Hough?

It is sensitive to lighting, shadows, glare, worn lane markings, and non-lane edges. It may detect road texture, vehicle edges, or shadows instead of actual lanes.

### Q21. Which model is best?

The safest answer is:
"It depends on the metric and scenario. In the lightweight notebook run, PolyFit performed best on the sampled mask-overlap metric because it directly uses lane-point geometry. However, for a real deployed system, segmentation is usually the more flexible direction, provided it is fully trained and validated. Classical Canny plus Hough is fast but less robust under difficult lighting and occlusion."

### Q22. Why not only use the best model?

Because model comparison is part of the methodology. A baseline tells us whether a more complex model is actually useful. Different models also fail in different ways, so comparing them gives a better understanding of the problem.

### Q23. Why do we need a baseline?

A baseline gives a minimum reference. If a complex model cannot beat a simple baseline, then the complex model is not justified.

### Q24. What is the difference between baseline and main model?

A baseline is a simple method used for comparison. A main model is the approach we would improve for final deployment. In our project, PolyFit and Canny/Hough are useful baselines, while segmentation is the stronger future direction.

## 6. Evaluation Q&A

### Q25. What metrics did you use?

We used accuracy, precision, recall, F1-score, and IoU-style mask overlap.

### Q26. Why not only use accuracy?

Accuracy can be misleading because most pixels are background, not lane pixels. A model can get high accuracy by predicting background correctly but still miss lane pixels. That is why precision, recall, F1, and IoU are important.

### Q27. What is precision in this project?

Precision measures how many predicted lane pixels are actually correct. High precision means the model does not draw many false lane markings.

### Q28. What is recall in this project?

Recall measures how many true lane pixels the model successfully detected. High recall means the model does not miss many lane markings.

### Q29. What is F1-score?

F1-score balances precision and recall. It is useful when both false positives and false negatives matter.

### Q30. What is IoU?

IoU means Intersection over Union. It compares the overlap between predicted lane mask and ground-truth lane mask. A higher IoU means the predicted lane area overlaps better with the true lane area.

### Q31. Why convert lanes to masks?

The annotation is a set of lane points, while different models produce different outputs: curves, segmentation masks, or line segments. Converting everything into binary masks gives a common evaluation format.

### Q32. Why is lane evaluation hard?

Because a lane is a thin spatial structure. A prediction can be visually close but still not overlap perfectly at the pixel level. Small shifts can hurt pixel metrics even if the lane looks reasonable.

### Q33. What were the main results?

From the latest notebook-generated findings:
- Best F1-score: PolyFit, around 0.6093
- Best mean IoU: PolyFit, around 0.4394
- Best precision: PolyFit, around 0.4518

Important explanation:
"These are from a lightweight sampled notebook evaluation, not a full CULane benchmark. We focus more on methodology, comparison, and error analysis than claiming state-of-the-art performance."

### Q34. Why did PolyFit perform well in the notebook run?

Because PolyFit works directly with lane geometry and the evaluation compares lane masks generated from lane coordinates. It is strong when the lane annotation is smooth and visible.

### Q35. Does that mean PolyFit is always best?

No. PolyFit can fail when lane boundaries are missing, occluded, or geometrically complex. A properly trained segmentation model would likely be stronger in many real-world difficult scenes.

## 7. Error Analysis Q&A

### Q36. Which scenarios are hardest?

The hardest scenarios are usually:
- No-line markings
- Night driving
- Crossroads/intersections
- Crowd/vehicle occlusion
- Strong glare or shadows

### Q37. Why are no-line markings difficult?

Because there may be little or no visible lane evidence. If the lane marking is missing, a model must infer lane position from road context, which is much harder.

### Q38. Why is night driving difficult?

At night, contrast is lower and headlights can create glare. Lane markings may not be clearly visible.

### Q39. Why are crossroads difficult?

Intersections can contain many road markings, multiple directions, and broken lane structure. The idea of a single clear lane boundary becomes ambiguous.

### Q40. Why is occlusion difficult?

Vehicles can cover the lane markings. The model must either infer the hidden lane or accept that part of the lane cannot be observed.

### Q41. Why do shadows cause problems?

Shadows create dark lines or edges that can look like lane markings to edge-based methods.

### Q42. How did you handle limitations?

We documented them through scenario-level analysis and visual examples. We did not hide failure cases. We explained where each method is weak and what could be improved.

### Q43. What failure cases did you observe or expect?

Common failure modes:
- False detection on shadows or road texture.
- Missed lanes under occlusion.
- Poor detection in faded/no-line scenes.
- Confusion at intersections.
- Straight-line methods failing on curves.

## 8. Code and Implementation Q&A

### Q44. What are the main project files?

- `Lane_Detection_Project.ipynb`: main presentation and code demo notebook.
- `train_pipeline.py`: model pipeline and supporting code.
- `visualize_annotations.py`: HTML visualization tool for annotation overlays.
- `outputs/`: generated figures, HTML visualizations, and findings.
- `data/CULane/`: dataset folder.

### Q45. How do we run the notebook?

Open `Lane_Detection_Project.ipynb` and run cells from top to bottom. The notebook uses portable paths, so it expects the dataset under `data/CULane` relative to the project folder.

### Q46. Which Python libraries did you use?

Main libraries:
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn
- SciPy
- Pandas
- pathlib and json from Python standard library

### Q47. Why OpenCV?

OpenCV is useful for image loading, resizing, color conversion, edge detection, Hough line detection, and drawing lane masks.

### Q48. Why Matplotlib?

Matplotlib is used for plotting images, overlays, metric charts, and scenario analysis figures.

### Q49. Why NumPy?

NumPy is used for arrays, image tensors, coordinates, mask operations, and metric calculations.

### Q50. Why is the path setup important?

The professor required portable paths. We avoid hardcoded laptop-specific paths and resolve the project root automatically. This helps the notebook run on another teammate's laptop.

### Q51. What does `LaneDataset` do?

It loads image paths, reads images, resizes them, normalizes pixel values, reads annotation files, and returns batches for evaluation.

### Q52. Why resize images?

Resizing standardizes input size and makes the notebook faster to run. Full-resolution images are large and slower to process.

### Q53. Does resizing affect accuracy?

Yes, it can. Resizing can slightly shift or blur thin lane markings. That is one reason the project is a lightweight demonstration rather than a final production benchmark.

### Q54. What outputs are generated?

The project generates:
- Model comparison figure.
- Scenario analysis figure.
- Results visualization figure.
- Findings text file.
- HTML annotation visualizations.

## 9. Professor-Style Critical Questions

### Q55. Why should we trust your results?

We trust the methodology more than a single score. We used official splits, visualized data, compared baselines, used common spatial metrics, and discussed limitations. We are careful not to claim state-of-the-art performance.

### Q56. What is the biggest weakness of your project?

The CNN is a reduced executable demonstration, not a fully trained production model. Also, pixel-based metrics can be harsh for thin lane lines. With more time, we would train a stronger segmentation model and evaluate using standard lane-detection benchmarks.

### Q57. If you had more time, what would you improve first?

First, train a proper segmentation network such as U-Net, ENet, SCNN-style architecture, or a modern lane-detection model. Second, evaluate on all official test splits. Third, tune preprocessing and post-processing for each scenario.

### Q58. How would you improve performance in night scenes?

Use night-specific augmentation, contrast enhancement, exposure normalization, and include more night examples during training. A model could also use temporal information from video frames.

### Q59. How would you improve performance under occlusion?

Use a segmentation model trained with occlusion cases, add data augmentation, and use temporal continuity from previous frames to infer hidden lane segments.

### Q60. Why did you not use a very advanced model?

Because the project focus is methodology, comparison, and explainability under a limited course setting. A very advanced model would require more compute, training time, and tuning. We chose models that are understandable and runnable for demonstration.

### Q61. Is this classification, regression, or segmentation?

It is mainly a computer vision lane-detection task. Depending on the model, it can be treated as regression of lane curves, segmentation of lane pixels, or line detection. Our project compares these different formulations.

### Q62. What is the input and output?

Input: dashcam road image.  
Output: predicted lane boundary, represented as a curve, line segments, or lane mask.

### Q63. Why is this a machine learning project if one model is classical CV?

Because the project compares multiple approaches, including learning-based segmentation. Classical CV is included as a baseline, which is good scientific practice.

### Q64. What is the role of validation data?

Validation data is used to check performance and compare models without touching the final test set. It helps avoid overfitting decisions to the test set.

### Q65. What is the role of test data?

Test data is for final generalization assessment. In CULane, test data is scenario-specific, so it helps understand performance under different road conditions.

### Q66. Did you use the test set for training?

No. The test set is separate and should not be used for training.

### Q67. Why is visual inspection part of methodology?

Because computer vision datasets can have label issues, hidden biases, and difficult cases. Visual inspection confirms that we understand the data and task.

### Q68. What does "weather and occlusion" mean in your title?

It refers to difficult visual conditions such as glare, shadow, night, crowded traffic, and partial blocking of lane markings. The title highlights robustness under non-ideal driving conditions.

### Q69. Are all difficult conditions literally weather?

Not all. Some are lighting, road geometry, or occlusion conditions. The phrase "weather and occlusion" is a broad project title for challenging visual road conditions.

### Q70. How is this useful in real life?

Lane detection is used in lane keeping, lane departure warning, driver assistance, and autonomous driving. Robustness matters because real roads are not always clean and well-lit.

## 10. Short Answers to Memorize

### Project title

"Lane and Road Boundary Detection Under Weather and Occlusion."

### Main question

"Which approach can detect lane boundaries most reliably under normal and difficult road conditions?"

### Dataset

"CULane, a dashcam lane-detection dataset with lane coordinate annotations and nine scenario-specific test categories."

### Methodology

"Visualization, EDA, leakage-safe split, model comparison, and error analysis."

### Models

"PolyFit baseline, CNN-style segmentation, and Canny plus Hough classical detection."

### Metrics

"Accuracy, precision, recall, F1-score, and IoU-style mask overlap."

### Main finding

"No lightweight method is perfect. PolyFit is strong and interpretable on sampled lane-geometry evaluation, classical methods are fast, and segmentation is the best direction for future robust systems."

### Biggest limitation

"The CNN is a reduced demonstration, not a fully trained production model."

### Best future improvement

"Train a stronger segmentation network and evaluate it on all official CULane test scenarios."

## 11. Team Q&A Ownership

Suggested ownership:

Niruta:
- Project motivation and problem statement.
- Dataset overview.
- Presentation introduction and conclusion.

Umesh:
- Methodology and split strategy.
- Data visualization and EDA.
- Why official split avoids leakage.

Ismail:
- Models, metrics, results, and error analysis.
- Limitations and future improvements.

Every team member should still be able to answer:
- What is the project about?
- What dataset did we use?
- Why is random split bad here?
- What are the three models?
- What are the main limitations?

## 12. Demo Checklist Before Presentation

Before presentation:
- Open `Lane_Detection_Project.ipynb`.
- Confirm the `Python 3 (Lane Detection)` kernel is working.
- Run at least the setup, dataset overview, and result cells once.
- Open `outputs/model_comparison_notebook.png`.
- Open `outputs/results_visualization_notebook.png`.
- Open one HTML annotation visualization from `outputs/`.
- Keep a backup copy of the notebook and outputs.
- Make sure all teammates have the final files.
- Check adapter/USB-C and internet availability.
- Practice explaining the project in one minute.

## 13. Honest Defense Lines

Use these if the professor challenges the work:

"We focused on a correct project workflow: understanding the data, avoiding leakage, comparing baselines, evaluating spatially, and analyzing errors."

"The CNN in this notebook is a reduced demonstration. We are not claiming it is a production-level trained model."

"Because lane markings are thin, pixel metrics are sensitive. That is why we combine metric tables with visual inspection."

"A random split would be risky because similar video frames could appear in both train and test. That is why we used the official split."

"Our final recommendation is not that one model solves every case. The main lesson is that lane detection is scenario-dependent, and robust systems need stronger segmentation, scenario handling, and careful evaluation."

## 14. Questions You Should Ask Each Other During Practice

1. Explain the project title in your own words.
2. What is CULane?
3. What is one image and one annotation in this dataset?
4. Why is this problem difficult?
5. What are the nine test scenarios?
6. Why did we avoid random splitting?
7. What is data leakage?
8. What does PolyFit do?
9. What does Canny plus Hough do?
10. What does segmentation mean?
11. Why are accuracy alone and pixel metrics sometimes misleading?
12. What is precision?
13. What is recall?
14. What is F1-score?
15. What is IoU?
16. Which scenario is hardest and why?
17. What is the biggest weakness of the project?
18. What would we improve with more time?
19. How would this help in real driving?
20. Why should the professor believe our methodology?
