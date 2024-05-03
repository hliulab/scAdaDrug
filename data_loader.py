from train_2 import *
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scAdaDrug_2 import *
from imblearn.over_sampling import SMOTE

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])



drug_sc = ['Etoposide', 'PLX4720', 'PLX4720_451Lu']
drug_tcga = ['cisplatin', 'docetaxel', 'fluorouracil', 'gemcitabine', 'paclitaxel', 'sorafenib']

drug = drug_sc[0]

# source_data = pd.read_csv('datasets/data_tcga/drug_'+drug_tcga[0]+'/3.'+drug+'_gdsc.csv')
# target_data = pd.read_csv('datasets/data_tcga/drug_'+drug_tcga[0]+'/'+drug+'_tcga.csv')
source_data = pd.read_csv('datasets/data_exp1/Source_exprs_resp_z.' + drug + '_tp4k.tsv', sep='\t')
target_data = pd.read_csv('datasets/data_exp1/Target_expr_resp_z.' + drug + '_tp4k.tsv', sep='\t')



# data_scad
source_x = source_data.iloc[:, 3:].values
source_y = source_data.iloc[:, 1].values



print("源域原始训练集中各类别的样本数量：")
print("类别 0: ", sum(source_y == 0))
print("类别 1: ", sum(source_y == 1))

# 使用SMOTE进行采样
smote = SMOTE(random_state=42)
source_x_resampled, source_y_resampled = smote.fit_resample(source_x, source_y)

print("源域采样后训练集中各类别的样本数量：")
print("类别 0: ", sum(source_y_resampled == 0))
print("类别 1: ", sum(source_y_resampled == 1))

source_x_train, source_x_test, source_y_train, source_y_test = train_test_split(source_x_resampled, source_y_resampled, test_size=0.3)

source_train_dataset = CustomDataset(source_x_train, source_y_train)
source_train_loader = DataLoader(source_train_dataset, batch_size=bs * 2, shuffle=True)
source_test_dataset = CustomDataset(source_x_test, source_y_test)
source_test_loader = DataLoader(source_test_dataset, batch_size=bs * 2, shuffle=True)

# data_scad
target_x = target_data.iloc[:, 2:].values
target_y = target_data.iloc[:, 1].values



print("目标域原始训练集中各类别的样本数量：")
print("类别 0: ", sum(target_y == 0))
print("类别 1: ", sum(target_y == 1))

# 使用SMOTE进行采样
smote = SMOTE(random_state=42)
target_x_resampled, target_y_resampled = smote.fit_resample(target_x, target_y)

print("目标域采样后训练集中各类别的样本数量：")
print("类别 0: ", sum(target_y_resampled == 0))
print("类别 1: ", sum(target_y_resampled == 1))
target_x_train, target_x_test, target_y_train, target_y_test = train_test_split(target_x, target_y, test_size=0.3)

target_train_dataset = CustomDataset(target_x_train, target_y_train)
target_train_loader = DataLoader(target_train_dataset, batch_size=bs, shuffle=True)

target_test_dataset = CustomDataset(target_x_test, target_y_test)
target_test_loader = DataLoader(target_test_dataset, batch_size=bs, shuffle=True)
