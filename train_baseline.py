import matplotlib.pyplot as plt
from baseline import *
from data_loader import *
from loss_and_metrics import *

device = 'cuda:1'
lr = 1e-5
bs = 64
# drug = 'crizotinib'
drug = 'sorafenib'

n_epochs = 300
num_columns = len(source_data.columns)

# Initialize model and optimizer
model = Model(device = device, ae_indim=num_columns-2, embedding_dim=256)
optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=0.001)
# optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)

for epoch in range(n_epochs):
    train_epoch_loss = 0
    train_epoch_reconstruction_loss = 0
    train_epoch_prediction_loss = 0
    train_epoch_adv_loss = 0
    model.train()

    y_true_train = []
    y_pred_train = []

    target_y_true_train = []
    target_y_pred_train = []
    target_y_true_all = []
    target_y_pred_all = []


    for i, data in enumerate(zip(cycle(source_train_loader), target_train_loader)):

        if data[0][0].size()!=data[1][0].size():
            continue
        sx = data[0][0].to(device)
        sy = data[0][1].to(device)
        tx = data[1][0].to(device)
        ty = data[1][1].to(device)

        s_encoded, s_decoded, s_pred, t_encoded, t_decoded, t_pred, domain_pre = model(sx, tx)

        Labels = torch.ones(bs, 1).to(device)
        Labelt = torch.zeros(bs, 1).to(device)
        domain_label = torch.cat([Labels, Labelt], 0).to(device)

        adv_losses = adv_loss(domain_pre, domain_label)
        reconstruction_losses = reconstruction_loss(s_decoded, sx)
        s_pred = s_pred.squeeze()
        prediction_losses = prediction_loss(s_pred, sy)
        total_loss = reconstruction_losses + prediction_losses + adv_losses

        train_epoch_loss += total_loss.item()
        train_epoch_reconstruction_loss += reconstruction_losses.item()
        train_epoch_prediction_loss += prediction_losses.item()
        train_epoch_adv_loss += adv_losses.item()

        y_true_train.extend(sy.cpu().detach().numpy())
        y_pred_train.extend(s_pred.cpu().detach().numpy())

        t_pre = t_pred.squeeze()
        target_y_true_train.extend(ty.cpu().detach().numpy())
        target_y_pred_train.extend(t_pre.cpu().detach().numpy())
        target_y_true_all.extend(ty.cpu().detach().numpy())
        target_y_pred_all.extend(t_pre.cpu().detach().numpy())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    train_total_loss.append(train_epoch_loss / len(target_train_loader))
    train_reconstruction_loss.append(train_epoch_reconstruction_loss / len(target_train_loader))
    train_prediction_loss.append(train_epoch_prediction_loss / len(target_train_loader))
    train_adv_loss.append(train_epoch_adv_loss / len(target_train_loader))

    # Evaluate model on testing data
    test_epoch_loss = 0
    test_epoch_reconstruction_loss = 0
    test_epoch_prediction_loss = 0
    test_epoch_adv_loss = 0
    y_true_test = []
    y_pred_test = []
    target_y_true_test = []
    target_y_pred_test = []
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(zip(cycle(source_test_loader), target_test_loader)):
            # print('test_batch', i, '-----------------------------------------')
            if data[0][0].size() != data[1][0].size():
                continue
            sx = data[0][0].to(device)
            sy = data[0][1].to(device)

            tx = data[1][0].to(device)
            ty = data[1][1].to(device)

            s_encoded, s_decoded, s_pred, t_encoded, t_decoded, t_pred, domain_pre = model(sx, tx)

            Labels = torch.ones(bs, 1).to(device)
            Labelt = torch.zeros(bs, 1).to(device)
            domain_label = torch.cat([Labels, Labelt], 0).to(device)

            adv_losses = adv_loss(domain_pre, domain_label)
            reconstruction_losses = reconstruction_loss(s_decoded, sx)
            s_pred = s_pred.squeeze()
            prediction_losses = prediction_loss(s_pred, sy)
            total_loss = reconstruction_losses + prediction_losses + adv_losses

            test_epoch_loss += total_loss.item()
            test_epoch_reconstruction_loss += reconstruction_losses.item()
            test_epoch_prediction_loss += prediction_losses.item()
            test_epoch_adv_loss += adv_losses.item()

            y_true_test.extend(sy.cpu().detach().numpy())
            y_pred_test.extend(s_pred.cpu().detach().numpy())

            t_pre = t_pred.squeeze()
            target_y_true_test.extend(ty.cpu().detach().numpy())
            target_y_pred_test.extend(t_pre.cpu().detach().numpy())
            target_y_true_all.extend(ty.cpu().detach().numpy())
            target_y_pred_all.extend(t_pre.cpu().detach().numpy())


        test_total_loss.append(test_epoch_loss / len(target_test_loader))
        test_reconstruction_loss.append(test_epoch_reconstruction_loss / len(target_test_loader))
        test_prediction_loss.append(test_epoch_prediction_loss / len(target_test_loader))
        test_adv_loss.append(test_epoch_adv_loss / len(target_test_loader))


    # AUROC (AUC-ROC)
    train_auroc = roc_auc_score(y_true_train, y_pred_train)
    # print('train_auroc:',train_auroc)
    test_auroc = roc_auc_score(y_true_test, y_pred_test)
    target_train_auroc = roc_auc_score(target_y_true_train, target_y_pred_train)
    target_test_auroc = roc_auc_score(target_y_true_test, target_y_pred_test)
    target_all_auroc = roc_auc_score(target_y_true_all, target_y_pred_all)

    train_aurocs.append(train_auroc)
    # print('train_aurocs:', train_aurocs)
    test_aurocs.append(test_auroc)
    target_train_aurocs.append(target_train_auroc)
    target_test_aurocs.append(target_test_auroc)
    target_all_aurocs.append(target_all_auroc)

    # AP
    train_ap = average_precision_score(y_true_train, y_pred_train)
    test_ap = average_precision_score(y_true_test, y_pred_test)
    target_train_ap = average_precision_score(target_y_true_train, target_y_pred_train)
    target_test_ap = average_precision_score(target_y_true_test, target_y_pred_test)
    target_all_ap = average_precision_score(target_y_true_all, target_y_pred_all)

    train_aps.append(train_ap)
    test_aps.append(test_ap)
    target_train_aps.append(target_train_ap)
    target_test_aps.append(target_test_ap)
    target_all_aps.append(target_all_ap)

    # Print progress
    print(f"Epoch {epoch + 1}: "
          f"Train Loss: {train_total_loss[-1]:.5f}, "
          f"Test Loss: {test_total_loss[-1]:.5f}, "
          f"Train AUC: {train_aurocs[-1]:.5f}, "
          f"Test AUC: {test_aurocs[-1]:.5f}, "
          f"Target Train AUC: {target_train_aurocs[-1]:.5f}, "
          f"Target Test AUC: {target_test_aurocs[-1]:.5f}, "
          f"Target All AUC: {target_all_aurocs[-1]:.5f}")

# Plot loss and auc curves
fig, axs = plt.subplots(3, 3, figsize=(15, 12))

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
axs[0][2].set_ylabel('Adv Loss')
axs[0][2].legend()

axs[1][0].plot(train_total_loss, label='Train')
axs[1][0].plot(test_total_loss, label='Test')
axs[1][0].set_xlabel('Epochs')
axs[1][0].set_ylabel('Total Loss')
axs[1][0].legend()

axs[1][1].plot(train_aurocs, label='Train')
axs[1][1].plot(test_aurocs, label='Test')
axs[1][1].set_xlabel('Epochs')
axs[1][1].set_ylabel('Source AUROC')
axs[1][1].legend()

axs[1][2].plot(target_train_aurocs, label='Train')
axs[1][2].plot(target_test_aurocs, label='Test')
axs[1][2].plot(target_all_aurocs, label='All')
axs[1][2].set_xlabel('Epochs')
axs[1][2].set_ylabel('Target AUROC')
axs[1][2].legend()

axs[2][0].plot(train_aps, label='Train')
axs[2][0].plot(test_aps, label='Test')
axs[2][0].set_xlabel('Epochs')
axs[2][0].set_ylabel('Source AUPR')
axs[2][0].legend()

axs[2][1].plot(target_train_aps, label='Train')
axs[2][1].plot(target_test_aps, label='Test')
axs[2][1].plot(target_all_aps, label='All')
axs[2][1].set_xlabel('Epochs')
axs[2][1].set_ylabel('Target AUPR')
axs[2][1].legend()

plt.tight_layout()
plt.show()

plt.close()
