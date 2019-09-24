# Kaggle: LANL Earthquake Prediction ([link](https://www.kaggle.com/c/LANL-Earthquake-Prediction/overview))

Data: 4194 segments of acoustic signals relating to laboratory earthquakes

Task: predict the time to failure (at the end of a segment)

Evaluation: Mean Absolute Error

Solution: Model stacking (1) LGB, (2) XGB, (3) NuSVR and (4) KRR

Success: 1.538 MAE

![](predictions.png)

![](feature_importance.png)

![](feature_importance_stack.png)
