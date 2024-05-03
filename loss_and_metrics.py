from scAdaDrug_2sources import *


# Define loss functions
reconstruction_loss = nn.MSELoss()
prediction_loss = nn.BCELoss()
adv_loss = nn.BCELoss()
ortho_loss = nn.MSELoss()

train_total_loss = []
train_reconstruction_loss = []
train_prediction_loss = []
train_adv_loss = []
train_ortho_loss = []

test_total_loss = []
test_reconstruction_loss = []
test_prediction_loss = []
test_adv_loss = []
test_ortho_loss = []

train_aurocs = []
test_aurocs = []
target_train_aurocs = []
target_test_aurocs = []
target_all_aurocs = []

train_aps = []
test_aps = []
target_train_aps = []
target_test_aps = []
target_all_aps = []

target_train_f1s = []
target_test_f1s = []
target_all_f1s = []
