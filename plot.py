from sklearn.metrics import precision_recall_curve
from train_2 import *
import matplotlib.pyplot as plt
import numpy as np

# AUPR
precision_train, recall_train, _train = precision_recall_curve(target_y_true_train, target_y_pred_train)
precision_test, recall_test, _test = precision_recall_curve(target_y_true_test, target_y_pred_test)

np.savetxt(drug + '_2source_smote_' + '_auroc.csv', target_test_aurocs, delimiter=',')

# Plot loss and auc curves
fig, axs = plt.subplots(3, 4, figsize=(20, 12))

axs[0][0].plot(train_reconstruction_loss, label='Train')
axs[0][0].plot(test_reconstruction_loss, label='Test')
axs[0][0].set_xlabel('Epochs')
axs[0][0].set_ylabel('Reconstruction Loss')
axs[0][0].legend()

axs[0][1].plot(train_prediction_loss, label='Train')
axs[0][1].plot(test_prediction_loss, label='Test')
axs[0][1].set_xlabel('Epochs')
axs[0][1].set_ylabel('Prediction Loss')
axs[0][1].legend()

axs[0][2].plot(train_adv_loss, label='Train')
axs[0][2].plot(test_adv_loss, label='Test')
axs[0][2].set_xlabel('Epochs')
axs[0][2].set_ylabel('adv Loss')
axs[0][2].legend()

axs[0][3].plot(train_ortho_loss, label='Train')
axs[0][3].plot(test_ortho_loss, label='Test')
axs[0][3].set_xlabel('Epochs')
axs[0][3].set_ylabel('ortho Loss')
axs[0][3].legend()

axs[1][0].plot(train_total_loss, label='Train')
axs[1][0].plot(test_total_loss, label='Test')
axs[1][0].set_xlabel('Epochs')
axs[1][0].set_ylabel('Total Loss')
axs[1][0].legend()

axs[1][1].plot(train_aurocs, label='Train')
axs[1][1].plot(test_aurocs, label='Test')
axs[1][1].set_xlabel('Epochs')
axs[1][1].set_ylabel('source AUC')
axs[1][1].legend()

axs[1][2].plot(train_aps, label='Train')
axs[1][2].plot(test_aps, label='Test')
axs[1][2].set_xlabel('Epochs')
axs[1][2].set_ylabel('source AP')
axs[1][2].legend()


axs[2][0].plot(target_train_aurocs, label='Train')
axs[2][0].plot(target_test_aurocs, label='Test')
axs[2][0].plot(target_all_aurocs, label='all')
axs[2][0].set_xlabel('Epochs')
axs[2][0].set_ylabel('target auc')
axs[2][0].legend()

axs[2][1].plot(target_train_aps, label='Train')
axs[2][1].plot(target_test_aps, label='Test')
axs[2][1].plot(target_all_aps, label='all')
axs[2][1].set_xlabel('Epochs')
axs[2][1].set_ylabel('target AP')
axs[2][1].legend()

axs[2][2].plot(recall_train, precision_train, marker='.')
axs[2][2].set_xlabel('Recall')
axs[2][2].set_ylabel('Precision')
axs[2][2].legend()

axs[2][3].plot(recall_test, precision_test, marker='.')
axs[2][3].set_xlabel('Recall')
axs[2][3].set_ylabel('Precision')
axs[2][3].legend()

# data_scad
fig.suptitle('drug:' + drug + '    ' + 'lr:' + str(lr), fontsize=16)
plt.tight_layout()
plt.savefig(drug+'_'+str(lr)+'_'+str(n_epochs)+'.png')



