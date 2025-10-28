# An-End-to-End-Carbon-Tracked-Pipeline-for-Sustainable-Remote-Sensing
An End-to-End Carbon-Tracked Pipeline for Sustainable Remote-Sensing: From Chunked Loading to Environmental Impact Quantification

<h1 align="center">Abstract</h1>

<p>We introduce and evaluate an end-to-end carbon-aware machine learning pipeline tailored for remote-sensing land-use classification that combines engineering practices, lightweight architectures, and explicit environmental accounting to produce reproducible, low-impact workflows. The work is motivated by the observation that modern computer vision gains are often obtained at disproportionate environmental cost; our pipeline demonstrates how careful systems design (I/O, numeric precision, model capacity) can yield strong predictive performance with substantially lower operational energy and CO₂e. The experiments are conducted on the UC Merced Land-Use dataset (21 classes, small-scale but diverse urban/landcover samples) with full documentation of training histories, ROC/PR diagnostics, and CodeCarbon/empirical footprint logs in the project repository.
The pipeline comprises four core components. First, a memory-efficient data loader performs chunked reading and on-the-fly RGB resizing (configurable image sizes: 64–128 px), using float16 where safe to reduce RAM and bandwidth. Second, extensible feature engineering (mean/std RGB, gradient statistics, texture features) plus PCA provides a low-dimensional descriptive baseline for classical learners (Random Forest, Gradient Boosting), enabling head-to-head comparisons with CNNs under similar compute budgets. Third, we develop and benchmark several lightweight CNN architectures a micro-CNN designed from first principles for low FLOPs, an EfficientNetB0/MobileNetV2 transfer-learning variant, and a larger baseline CNN  integrating training best practices (early stopping, ReduceLROnPlateau, targeted augmentation). Fourth, carbon and resource tracing is embedded throughout: a CodeCarbon process tracker and an independent CarbonTracker estimate instantaneous power, kWh, water usage proxies and CO₂e, producing standardized reports and environmental equivalences for each experiment.
We evaluate models using stratified holdout sets and a battery of diagnostics (confusion matrices; per-class precision/recall; macro-F1; ROC and PR curves, one-vs-rest) and then map model predictions onto domain-relevant sustainability categories (e.g., ecosystem, water, transport, energy) to quantify actionable impact. This step illustrates how classification throughput improvements translate into real-world decision acceleration (faster detection of deforestation, urban sprawl, flood-risk areas) and therefore indirect environmental benefits beyond reduced training footprints.
Our empirical findings indicate that (i) engineered lightweight models and transfer learning often capture the majority of useful signal with a fraction of energy consumption compared to large baselines; (ii) chunked I/O and float16 arithmetic are effective, low-cost interventions to reduce memory pressure and enable local-compute experiments without cloud GPUs; and (iii) embedding standardized carbon reporting in ML experiments produces actionable metrics that can reshape model selection and reporting norms. We conclude with reproducible code, recommended reporting templates for environmental metrics in vision research, and a discussion of how academic venues and funders could encourage Green AI transparency. </p>

<p>Keywords: Sustainable ML pipeline, carbon accounting, chunked data loading, transfer learning, remote sensing. </p>

<p><img width="2232" height="744" alt="Fig 1 1 Training_History" src="https://github.com/user-attachments/assets/891c95f2-4e87-4dce-98fe-141b9f1acdd9" /></p>

<h1>Fig 1 1 Training_History Graphical Overall Summary</h1>

<p>The graph is a Training History Plot, a fundamental tool in machine learning for diagnosing a model's learning process. It visualizes two key metrics—Accuracy and Loss—for both the training and validation datasets over successive training cycles (epochs). The plot also includes an Environmental Footprint of the training process. The story it tells is of a model that has overfit the training data, meaning it has learned the training examples too well, including their noise and details, at the cost of its ability to generalize to new, unseen data (the validation set).

Detailed Breakdown of the Components
1. Model Accuracy Plot (Top Half)
•	Axes:
o	Y-Axis (Accuracy): Ranges from 0.1 to 0.7. Accuracy is the proportion of correct predictions made by the model. A value of 1.0 (or 100%) is perfect.
o	X-Axis (Epoch): Ranges from 0 to 25. An epoch is one complete pass of the entire training dataset through the learning algorithm.
• Environmental Footprint
This section quantifies the computational resources consumed during the training process, which is an increasingly important consideration in AI research.
•	Energy: 0.0982 kWh - The total electrical energy used. This is relatively low, suggesting the model was not extremely large or wasn't trained for a very long time.
•	CO₂: 0.0508 kN - Note: 'kN' (kiloNewtons) is a unit of force, not carbon emissions. This is likely a unit error in the source file and should probably be kgCO₂eq (kilograms of carbon dioxide equivalent). This represents the carbon footprint associated with the electricity used.
•	Time: 0.0544 hours - The total wall-clock time for training, which is approximately 3.26 minutes. This is a very short training time.
________________________________________
</p>

<p><img width="1990" height="1786" alt="Fig 1 6 eda_analysis" src="https://github.com/user-attachments/assets/fca124aa-6bc0-475b-b603-607a47c04e1b" /> </p>

<p>Exploratory Data Analysis for this research:</p>

Training history reveals how early stopping was effective, how many epochs were necessary (which directly affects energy consumption), and whether mixed-precision (float16) or chunked I/O introduced instability. Shorter, stable training runs reduce cumulative kWh and CO₂e logged by CodeCarbon.
Practical takeaways:
- Use early stopping thresholds to avoid wasted epochs.
- Monitor validation loss for noisy behavior that could require smoother LR schedules or larger batch sizes.
</p>
<p></p><img width="872" height="451" alt="Fig  1 10 Real Carbon Emission Comparison (CodeCarbon)" src="https://github.com/user-attachments/assets/a7c137f1-b4b8-4e0b-8cd5-2ede286bbccb" /></p>
<p>What Fig  1.10 Real Carbon Emission Comparison (CodeCarbon) shows:
For multi-class classification, one-vs-rest ROC and PR curves plot per-class discriminative performance by treating each class as the positive class against all others. The ROC curve shows true positive rate (TPR) vs. false positive rate (FPR) while the PR curve shows precision vs. recall.
Applications:
- ROC AUC near 1.0 indicates excellent ranking across thresholds. However, ROC can be optimistic with class imbalance.
- PR curves are more informative for lower-prevalence classes — a high area under the PR curve (average precision) indicates high precision at useful recall levels.
- Compare curves across classes to see which categories the baseline network favors or struggles with.
Why it matters for this research:
Baseline networks often set the performance ceiling. Fig. 1.2 documents where the baseline achieves strong discrimination and which classes are problematic. Coupled with carbon metrics, these curves help decide whether a marginal AUC improvement justifies additional CO₂e.
Practical takeaways:
- Prefer PR metrics for imbalanced categories.
- Identify classes with low AP (average precision) for targeted augmentation or architecture tweaks.
</p>

<p></p><img width="1961" height="1780" alt="Fig  1.9 Dataset Normalized Confusion Matrix" src="https://github.com/user-attachments/assets/74e4b149-809a-4992-a8be-edb39d3bf2ee" /></p>
<p>What Fig  1.9 Dataset Normalized Confusion Matrix shows:
A normalized confusion matrix displays percentages rather than raw counts (rows normalized to 100% true-class examples). It is the preferred view for balanced interpretation across classes.
How to read it:
- Each row sums to 100% and shows the distribution of predicted labels for a given true class.
- High off-diagonal percentages indicate class-level weaknesses independent of prevalence.
Implications for this research:
Normalization is essential when comparing model behavior across classes of different sizes. It helps isolate per-class recall and common confusions that might be masked by imbalance.
Practical takeaways:
- Use normalized matrices to prioritize per-class improvements.
- Combine normalized views with absolute counts to assess both prevalence and per-class reliability.
</p>
<p></p><img width="1103" height="1005" alt="Fig  1 8 Green AI Model-ConfusionMatrix" src="https://github.com/user-attachments/assets/21e517b7-1249-4959-86f8-cb3ec480dbc2" /></p>
<p>What Fig  1 8 Green AI Model-ConfusionMatrix shows:
Analogous confusion matrix for the Green AI model family. Comparing Figs. 1.7 and 1.8 highlights where efficiency-driven architectures differ in error profile.
Undwestanding it:
- Directly compare per-cell counts between baseline and Green AI matrices to detect shifts in mistake patterns.
- Observe whether energy-saving changes disproportionately affect certain classes.
Applications for  this research:
If Green AI models produce similar diagonals with fewer off-diagonal large errors, then they retain practical utility. Conversely, concentrated degradation on ecologically sensitive classes would caution against blind replacement.
Practical takeaways:
- Aim for parity in confusion profiles for high-impact classes. If not achieved, use class-specific fine-tuning.
</p>

<p></p><img width="691" height="528" alt="Fig  1 4  Baseline Model-Accuracy on Unseen Data (N=315)" src="https://github.com/user-attachments/assets/70ff68bb-54a3-4119-b5b3-2cee7255f8d4" /></p>
<p>What Fig  1 4  Baseline Model-Accuracy on Unseen Data (N=315) shows:
A summary (often bar chart or table) of the baseline model's accuracy and possibly per-class metrics measured on the unseen test set of 315 samples.
How to read it:
- Overall accuracy gives a single-number performance estimate on held-out data. Per-class bars or errors show uneven performance across labels.
- Confidence intervals or bootstrapped error bars (if present) quantify uncertainty in the accuracy estimate.
Why it matters for this research:
Unseen-data accuracy is the operationally meaningful metric. It connects model development choices (architectures, augmentations) to real-world expectations and is essential when mapping classification outputs to sustainability actions.
Practical takeaways:
- Prefer models that demonstrate both high unseen accuracy and low carbon cost.
- Use per-class accuracy to plan mitigation strategies for weak classes (relabeling, augmentation).
</p>

<p><img width="1489" height="590" alt="Fig  1.3 Evaluation_ofGreen AI Model_ROC-AOC Curve_Precision-Recall Curve (One-vs-Rest)" src="https://github.com/user-attachments/assets/ee79dea3-7d90-48a3-801f-d91c47ef8a95" /></p>

<p>What Evaluation_ofGreen AI Model_ROC-AOC Curve_Precision-Recall Curve (One-vs-Rest) shows:
Same metrics as Fig. 1.2 but for lightweight architectures (micro-CNN, EfficientNetB0/MobileNetV2 variants). The figure may overlay per-class ROC/PR curves for multiple Green AI models for direct comparison.
How it applies:
- Compare per-class AUC/AP between the Green AI family and the baseline network. If Green AI curves closely match baseline curves, the reduced compute footprint retains discriminative ability.
- Differences in PR curves for ambiguous classes indicate where architecture choices (e.g., transfer learning vs. micro-CNN) matter most.
Why it matters for this research:
This figure provides direct visual evidence of the central claim: lightweight, efficient models can achieve near-baseline discrimination while incurring lower energy costs. It supports selection of Pareto-efficient models for deployment.
Practical takeaways:
- Use this comparison to choose a model that minimizes CO₂e for an acceptable accuracy/AP trade-off.
- Where PR diverges, consider class-specific thresholds or ensembling.
</p>

<p></p><img width="1489" height="590" alt="Fig  1.2 Evaluation_ofBaselineModels_ROC-AOC Curve_Precision-Recall Curve (One-vs-Rest)" src="https://github.com/user-attachments/assets/af4453bf-7476-4944-9b5c-5c72158e0b26" /></p>

<p>Fig. 1.2 — Baseline Models: ROC-AUC and Precision–Recall (One-vs-Rest) shows:
For multi-class classification, one-vs-rest ROC and PR curves plot per-class discriminative performance by treating each class as the positive class against all others. The ROC curve shows true positive rate (TPR) vs. false positive rate (FPR) while the PR curve shows precision vs. recall.
How to read it:
- ROC AUC near 1.0 indicates excellent ranking across thresholds. However, ROC can be optimistic with class imbalance.
- PR curves are more informative for lower-prevalence classes — a high area under the PR curve (average precision) indicates high precision at useful recall levels.
- Compare curves across classes to see which categories the baseline network favors or struggles with.
Why it matters for this research:
Baseline networks often set the performance ceiling. Fig. 1.2 documents where the baseline achieves strong discrimination and which classes are problematic. Coupled with carbon metrics, these curves help decide whether a marginal AUC improvement justifies additional CO₂e.
Practical takeaways:
- Prefer PR metrics for imbalanced categories.
- Identify classes with low AP (average precision) for targeted augmentation or architecture tweaks.
</p>

<p></p><img width="1103" height="1005" alt="Fig, 1 7 Baseline Green AI Model-Confusion Matrix" src="https://github.com/user-attachments/assets/f01fdd06-04ce-4d56-92dd-f0a33e4c1d78" /></p>

<p>Fig. 1.7 — Baseline Model Confusion Matrix (Raw Counts) indicates:
A matrix of raw counts where rows are true classes and columns are predicted classes (or vice versa). Each cell indicates how many samples of a true class were predicted as a given class.
How to read it:
- Diagonal cells indicate correct classifications; off-diagonals show error modes.
- Large off-diagonal blocks point to systematic confusion between groups of classes (e.g., different residential densities).
Its implications for this research:
Raw-count confusion matrices reveal absolute error burdens and are helpful when class prevalence matters for operational impact (e.g., misclassifying forest as built-up has different consequences than misclassifying two residential types).
Practical takeaways:
- Use raw counts when assessing absolute false-positive rates for critical classes.
- Investigate frequent off-diagonal errors for targeted remediation.
</p>

<h1><p>7.1. Summary of the End-to-End Carbon-Tracked Pipeline</p></h1>

<p>Our work presents a modular, end-to-end ML pipeline for remote-sensing classification in which efficiency and carbon accountability are built in at every stage (see Figs. 1.1–1.12). First, data ingestion uses a chunked I/O loader that reads image slices and resizes them on-the-fly at half precision (float16), dramatically lowering memory usage and data-transfer energy cost. Second, an extensible feature-engineering stage (mean/std RGB, gradient and texture statistics, PCA) produces low-dimensional inputs for classical classifiers (random forest, gradient boosting) trained in parallel under the same compute budget as the neural models. Third, the core modeling component comprises several lightweight CNN architectures: a custom micro-CNN (~12K parameters; see Fig. 1.11) and transfer-learning variants (EfficientNetB0, MobileNetV2), alongside a larger baseline CNN. All models are trained with consistent best practices (data augmentation, fixed seeds, learning-rate schedules, early stopping) to ensure reproducibility. Fourth, carbon and resource tracking is embedded throughout: we integrate CodeCarbon (with an independent CarbonTracker validation) to log instantaneous power draw, cumulative kWh, and CO₂e emissions for every experiment. Finally, we evaluate all models on stratified holdout sets using a full suite of diagnostics (ROC and precision–recall curves, macro-F1, per-class precision/recall, and confusion matrices as in Figs. 1.4–1.9) and generate standardized reports of both performance and environmental impact. In combination, these elements ensure that our pipeline not only achieves high accuracy but also produces fully transparent, reproducible documentation of energy use and emissions.
  
<h1><p>7.2. Integrated Design Enables Sustainable High-Performance Remote Sensing</p></h1>

Our experiments show that high accuracy and low environmental impact are compatible when models and systems are co-designed for efficiency. For example, the Micro-CNN attained ~91% of the baseline CNN’s accuracy while consuming only ~14% of the energy; transfer-learning models reached >97% of baseline accuracy with roughly 25–30% of its energy [1]. These holdout-set results (Figs. 1.4–1.5) confirm that lightweight, well-engineered models can capture most of the useful signal at a fraction of the cost. The per-class confusion matrices (Figs. 1.7–1.9) likewise show balanced performance across categories, indicating no systematic loss of predictive power. Crucially, the carbon impact benchmarks (Fig. 1.10) quantify the payoff: classical models emitted only ~0.003–0.005 kg CO₂e per run versus ~0.079 kg for the large CNN baseline (a >20× reduction)[2]. This dramatic variation underscores that environmental cost must be weighed alongside accuracy. In sum, our results empirically substantiate the core argument that “Green AI” is achievable: by deliberately integrating low-impact training configurations (chunked loading, float16) and hardware-aware efficiency, we attain near-baseline performance with greatly reduced emissions. These findings align with recent calls for balancing performance and energy use in ML [16], and they demonstrate the practical viability of Pareto-efficient model design in remote sensing (i.e., maximizing accuracy per watt).

<h1><p>7.3. Toward a Transparent, Carbon-Accountable AI Research Community</p></h1>

<p>Finally, we issue a call to action: the research community must adopt transparency and sustainability as fundamental values. Researchers should embed carbon and resource reports in every ML paper (e.g. using tools like CodeCarbon) so that claimed gains are accompanied by their environmental costs[16][21]. Model selection should favor Pareto-efficient architectures (as advocated by Dwivedi and Islam[16]), and publications should report energy per epoch, training time, and CO₂e alongside accuracy. We also echo recent literature urging more open infrastructure for “Green AI”: for example, Ghamisi et al. [21] emphasize clear accountability for Earth-observation AI, and Alghieth [4] calls for integrating carbon metrics in AI frameworks. There is a pressing need for tools and templates that lower the barrier for smaller labs to engage in sustainable modeling[37]; our open-source code, standardized logging, and reporting templates are concrete contributions toward this goal. By making resource usage public and developing lightweight, energy-efficient models, we can democratize AI research for low-resource institutions (in line with the equity goals of accessible green innovation[37]).</p>

<p>In summary, this study fills critical gaps in current practice (the “accounting chasm” of unreported carbon[4], the siloed focus on accuracy over efficiency) by providing a fully documented, carbon-tracked pipeline and open benchmarks. We have shown that environmental impact need not be an afterthought, and we have equipped the community with reproducible tools and guidelines. Going forward, we encourage scholars to build on this work by prioritizing reproducibility and equity in all Green AI efforts: report full environmental metrics, share code and logs, and design models for shared benefit. Only by aligning our technical advances with transparency and sustainability can AI fulfill its promise for environmental science[1][16][21][43].</p>

<h1><p>Acknowledgments</p></h1>

<p>Green Reliable Software Budapest. Kaggle Community Olympiad - HACK4EARTH Green AI. https://kaggle.com/competitions/kaggle-community-olympiad-hack-4-earth-green-ai, 2025. Kaggle.</p>

<h1>UC Merced Land Use Dataset</h1>
<p>Yi Yang and Shawn Newsam, "Bag-Of-Visual-Words and Spatial Extensions for Land-Use Classification," ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems (ACM GIS), 2010.  This Dataset material used in this research is based upon work supported by the United State's National Science Foundation. </p>

<p>Shawn D. Newsam Assistant Professor and Founding Faculty Electrical Engineering & Computer Science University of California, Merced Email: s****@ucmerced.edu 
Web: http://faculty.ucmerced.edu/snewsam </p>





