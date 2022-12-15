"""Anomaly metrics."""
import numpy as np
from sklearn import metrics
import pandas as pd

def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
##########
#     print(np.expand_dims(np.array(anomaly_ground_truth_labels).astype('int'),1) ,np.expand_dims(anomaly_prediction_weights,1))
    # anomaly_prediction_weights_2=np.expand_dims(anomaly_prediction_weights,1)
    # df_ano_score =pd.DataFrame(anomaly_prediction_weights_2)
    # df_ano_score.to_csv("anomaly_scores.csv")
    # anomaly_ground_truth_labels_2=np.expand_dims(np.array(anomaly_ground_truth_labels).astype('int'),1)
    # df_gt_labels =pd.DataFrame(anomaly_ground_truth_labels_2)
    # df_gt_labels.to_csv("gt.csv")
#     #x_train, x_test, y_train, y_test=train_test_split(anomaly_prediction_weights_2,anomaly_ground_truth_labels_2)
#     import tensorflow as tf
#     from tensorflow.keras.models import Sequential
#     from tensorflow.keras.layers import Dense, Activation
#     import pandas as pd
#     #import numpy as np

#     # Use numpy arrays to store inputs (x) and outputs (y):
#     #x = np.array([[0,0], [0,1], [1,0], [1,1]])
#     #y = np.array([[0], [1], [1], [0]]) 

#     # Define the network model and its arguments. 
#     # Set the number of neurons/nodes for each layer:
#     model = Sequential()
#     model.add(Dense(2, input_shape=(1,)))
#     model.add(Activation('relu'))
#     model.add(Dense(1))
#     model.add(Activation('sigmoid')) 

#     # Compile the model and calculate its accuracy:
#     opt =tf.keras.optimizers.SGD(
#     learning_rate=0.01,
#     momentum=0.0,
#     nesterov=False,
#     name='SGD',
    
# )
    # model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy']) 

    # # Print a summary of the Keras model:
    # model.summary()
    # model.fit(anomaly_prediction_weights_2 ,anomaly_ground_truth_labels_2,epochs=10,validation_split=0.3)

##########


    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": thresholds}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }
