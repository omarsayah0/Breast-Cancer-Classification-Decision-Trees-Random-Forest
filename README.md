# Breast Cancer Classification using Decision Tree & Random Forest

## About
This project implements Decision Tree and Random Forest machine learning models to classify breast cancer tumors as Malignant or Benign using the Breast Cancer Wisconsin dataset.
It includes data preprocessing, hyperparameter tuning with GridSearchCV, and comprehensive model evaluation using classification reports, confusion matrices, and ROC-AUC curves.
Additionally, the project provides visualizations for the decision tree structure, feature importance rankings, and comparative performance analysis between both algorithms to ensure a reliable and interpretable predictive model.

---

## Steps Included

### 1️⃣ Data Preprocessing
- Loaded the Breast Cancer Wisconsin dataset using sklearn.datasets.load_breast_cancer().

- Converted the dataset into a structured pandas.DataFrame with all feature names.

- Separated features (X) and target labels (y):

- X → all numeric diagnostic measurements

- y → binary target labels (0 = Malignant, 1 = Benign)

---

### 2️⃣ Model Training
- Built an **Decision Tree and Random Forest** model using `DecisionTreeClassifier` and `RandomForestClassifier` from `scikit-learn`.  
- Performed hyperparameter tuning with GridSearchCV to identify the optimal parameters for each model.
  
    1- For the Decision Tree, tuned parameters such as criterion, max_depth, and min_samples_split.

    2- For the Random Forest, optimized parameters including n_estimators, max_depth, criterion, and min_samples_split.

---


## How to Run

1- Install Dependencies:
  ```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

2-Run :

  ```bash
python decistion_tree_random_forest.py
```

3- Model Evaluation :

- The figure below compares the Decision Tree (left) and Random Forest (right) models using classification report heatmaps.
Each heatmap displays three key metrics Precision, Recall, and F1-score for both cancer classes:

      0 → Malignant (cancerous)

      1 → Benign (non-cancerous)

      Interpretation:

      Both models achieved high performance across all metrics, with overall accuracies around 96%.

      The Decision Tree model shows a slight drop in recall (0.91) for the malignant class, indicating a few false negatives (missed cancer cases).

      The Random Forest model improved recall (0.93) and maintained excellent precision (0.98), showing its ability to generalize better.

      The F1-scores for both classes remain consistently high (≈ 0.96–0.97), confirming strong balance between precision and recall.

      The Random Forest provides a more stable and robust performance than the single Decision Tree, thanks to ensemble averaging — reducing overfitting and improving     sensitivity to malignant cases.


<p align="center">
<img width="1518" height="746" alt="Capture_2025_11_08_12_59_37_594" src="https://github.com/user-attachments/assets/692afe79-a1ba-449e-ba16-33ca442e6c45" />
</p>


- The figure below compares the Decision Tree (left) and Random Forest (right) models using their confusion matrices.
Each matrix illustrates how many samples were correctly or incorrectly classified into the two categories:

      0 → Malignant (cancerous)

      1 → Benign (non-cancerous)

      Interpretation:

      The Decision Tree correctly classified 39 malignant and 70 benign cases but misclassified 4 malignant tumors as benign and 1 benign tumor as malignant.

      The Random Forest model improved slightly, with only 3 malignant cases misclassified as benign and 1 benign case misclassified as malignant.

      The diagonal cells (from top-left to bottom-right) represent correct predictions, showing that both models achieved very high accuracy and minimal classification errors.

      The Random Forest demonstrates better generalization and slightly higher sensitivity toward detecting malignant cases, making it a more reliable model for medical diagnostic prediction compared to a single Decision Tree.


<p align="center">
<img width="1560" height="697" alt="Capture_2025_11_08_12_59_42_511" src="https://github.com/user-attachments/assets/1bd71192-0589-4382-995a-813b2dc075e6" />
</p>


- The figure below displays the (ROC) curves for both models. Decision Tree (blue) and Random Forest (orange).
The ROC curve plots the True Positive Rate (Sensitivity) against the False Positive Rate, providing a visual assessment of how well each model distinguishes between the two classes:

      0 → Malignant (cancerous)

      1 → Benign (non-cancerous)

      Interpretation:

      The Decision Tree achieved an AUC (Area Under Curve) of 0.95, indicating strong classification performance.

      The Random Forest achieved a perfect AUC of 1.00, showing flawless separation between malignant and benign cases on the test set.

      The closer the ROC curve approaches the top-left corner, the better the model is at minimizing false positives while maximizing true positives.

      The Random Forest’s near-perfect ROC curve demonstrates its superior generalization ability and robustness, confirming that ensemble methods can significantly improve predictive accuracy over a single decision tree.
      This highlights its potential reliability for medical diagnostic applications, where sensitivity and accuracy are critical.


<p align="center">
<img width="723" height="528" alt="Capture_2025_11_08_12_59_52_387" src="https://github.com/user-attachments/assets/7b18ba41-11c4-4fec-8979-726df21fe8ba" />
</p>


- The figures below illustrate the structure of the trained Decision Tree and one individual tree from the Random Forest model.
  
      Each node in the tree represents a decision rule based on a specific feature threshold (e.g., mean concave points ≤ 0.05).
      The color of each node indicates the predicted class(
      🟦 Benign and 🟧 Malignant) while the intensity shows how pure the node is (how confident the model is about its prediction).

      Interpretation:

      The Decision Tree shows the full decision logic, clearly visualizing how the model splits data step-by-step using key features such as mean concave points, worst radius, and worst perimeter.

      Each branch narrows down the decision, and leaf nodes display final predictions with class labels, sample counts, and entropy values.

      The Random Forest visualization displays one of the many decision trees in the ensemble.
      Even though each tree may look simpler or slightly different, the forest as a whole combines their outputs to achieve higher stability and accuracy.

      Features like mean concave points, worst radius, and worst texture appear frequently in both models, confirming their strong diagnostic importance.

      The Random Forest benefits from aggregating multiple trees (reducing overfitting and improving robustness) while maintaining interpretability through these visualizations.


<p align="center">
<img width="48%" height="1316" alt="Capture_2025_11_08_13_00_03_999" src="https://github.com/user-attachments/assets/c33dfd3a-c92e-4708-8f5b-eeaaa322784e" />
<img width="48%" height="1341" alt="Capture_2025_11_08_13_00_09_642" src="https://github.com/user-attachments/assets/c867e49e-2f1f-4aaf-a521-83a4c0a93eaa" />
</p>

- The figure below illustrates the feature importance scores generated by the Random Forest model for the breast cancer classification task.
  
      Each bar represents how much a particular feature contributes to the model’s decision making process features with higher scores have a greater influence on predictions.

      Interpretation:
      
      The most important predictors are worst perimeter, worst area, worst concave points, and mean concave points.
      
      These features capture geometric and texture related properties of the tumor that are strongly correlated with malignancy.
      
      Lower-ranked features, such as smoothness error and mean symmetry, contribute less and may be considered less critical for prediction.
      
      The descending order of feature importance demonstrates how the Random Forest model automatically identifies and prioritizes the most relevant medical indicators for distinguishing between malignant and benign tumors.
      
      Feature importance visualization helps in model interpretability, providing transparency about which biological and structural tumor characteristics most influence the diagnosis.
      This is especially valuable in medical decision support systems, where understanding why the model made a prediction is as crucial as the prediction itself.

  
<p align="center">
<img width="1108" height="728" alt="Capture_2025_11_08_13_00_19_827" src="https://github.com/user-attachments/assets/557e3f82-92df-4915-9e46-b95f7aa44346" />
</p>



 ## Author
  
  Omar Alethamat

  LinkedIn : https://www.linkedin.com/in/omar-alethamat-8a4757314/

  ## License

  This project is licensed under the MIT License — feel free to use, modify, and share with attribution.
