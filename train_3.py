from data_loader import *
from loss_and_metrics import *
from sklearn.metrics import roc_auc_score, average_precision_score
from itertools import cycle
from scAdaDrug_3 import *

device = 'cuda:1'
lr = 6e-5
bs = 32
n_epochs = 400

num_columns = len(source_data.columns)

# data_scad
model = Model(device=device, ae_indim=num_columns - 3, embedding_dim=256)

# data_precily
# model = Model(device=device, ae_indim=num_columns - 2, embedding_dim=128)

optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=0.001)


eye = torch.zeros((bs), device=device)

for epoch in range(n_epochs):
    train_epoch_loss = 0
    train_epoch_reconstruction_loss = 0
    train_epoch_prediction_loss = 0
    train_epoch_adv_loss = 0
    train_epoch_ortho_loss = 0
    model.train()

    y_true_train = []
    y_pred_train = []

    target_y_true_train = []
    target_y_pred_train = []
    target_y_true_all = []
    target_y_pred_all = []

    for i, data in enumerate(zip(source_train_loader, cycle(target_train_loader))):
        if (data[0][0].size()[0] != bs * 3) | (data[1][0].size()[0] != bs):
            continue

        sx = data[0][0].to(device)

        s1x = sx[::3]
        s2x = sx[1::3]
        s3x = sx[2::3]

        ssx = torch.cat((s1x, s2x, s3x), dim=1)

        sy = data[0][1].to(device)
        s1y = sy[::3]
        s2y = sy[1::3]
        s3y = sy[2::3]
        sy = torch.cat((s1y, s2y, s3y), dim=0)

        tx = data[1][0].to(device)
        ty = data[1][1].to(device)

        corr, s1_encoded, s1_decoded, s1_pred, s2_encoded, s2_decoded, s2_pred, s3_encoded, s3_decoded, s3_pred, t_encoded, t_decoded, t_pred, domain_pre = model(
            s1x, s2x, s3x, tx)

        Labels = torch.ones(bs * 3, 1).to(device)
        Labelt = torch.zeros(bs, 1).to(device)
        domain_label = torch.cat([Labels, Labelt], 0).float().to(device)

        ortho_losses = ortho_loss(corr, eye)
        adv_losses = adv_loss(domain_pre, domain_label)
        reconstruction_losses = reconstruction_loss(s1_decoded, s1x) + reconstruction_loss(s2_decoded, s2x) + reconstruction_loss(s3_decoded, s3x)
        s1_pred = s1_pred.squeeze()
        s2_pred = s2_pred.squeeze()
        s3_pred = s3_pred.squeeze()
        s_pred = torch.cat((s1_pred, s2_pred, s3_pred), dim=0)
        prediction_losses = prediction_loss(s1_pred, s1y) + prediction_loss(s2_pred, s2y) + prediction_loss(s3_pred,
                                                                                                            s3y)
        total_loss = reconstruction_losses + prediction_losses + adv_losses + ortho_losses

        train_epoch_loss += total_loss.item()
        train_epoch_reconstruction_loss += reconstruction_losses.item()
        train_epoch_prediction_loss += prediction_losses.item()
        train_epoch_adv_loss += adv_losses.item()
        train_epoch_ortho_loss += ortho_losses.item()

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
    train_ortho_loss.append(train_epoch_ortho_loss / len(target_train_loader))

    # Evaluate model on testing data
    test_epoch_loss = 0
    test_epoch_reconstruction_loss = 0
    test_epoch_prediction_loss = 0
    test_epoch_adv_loss = 0
    test_epoch_ortho_loss = 0
    y_true_test = []
    y_pred_test = []
    target_y_true_test = []
    target_y_pred_test = []
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(zip(source_test_loader, cycle(target_test_loader))):
            if (data[0][0].size()[0] != bs * 3) | (data[1][0].size()[0] != bs):
                continue

            sx = data[0][0].to(device)
            s1x = sx[::3]
            s2x = sx[1::3]
            s3x = sx[2::3]
            ssx = torch.cat((s1x, s2x, s3x), dim=1)

            sy = data[0][1].to(device)
            s1y = sy[::3]
            s2y = sy[1::3]
            s3y = sy[2::3]
            sy = torch.cat((s1y, s2y, s3y), dim=0)

            tx = data[1][0].to(device)
            ty = data[1][1].to(device)

            corr, s1_encoded, s1_decoded, s1_pred, s2_encoded, s2_decoded, s2_pred, s3_encoded, s3_decoded, s3_pred, t_encoded, t_decoded, t_pred, domain_pre = model(
                s1x, s2x, s3x, tx)

            Labels = torch.ones(bs * 3, 1).to(device)
            Labelt = torch.zeros(bs, 1).to(device)
            domain_label = torch.cat([Labels, Labelt], 0).float().to(device)

            ortho_losses = ortho_loss(corr, eye)
            adv_losses = adv_loss(domain_pre, domain_label)
            reconstruction_losses = reconstruction_loss(s1_decoded, s1x) + reconstruction_loss(s2_decoded,
                                                                                               s2x) + reconstruction_loss(
                s3_decoded, s3x)
            s1_pred = s1_pred.squeeze()
            s2_pred = s2_pred.squeeze()
            s3_pred = s3_pred.squeeze()
            s_pred = torch.cat((s1_pred, s2_pred, s3_pred), dim=0)
            prediction_losses = prediction_loss(s1_pred, s1y) + prediction_loss(s2_pred, s2y) + prediction_loss(s3_pred,
                                                                                                                s3y)
            total_loss = reconstruction_losses + prediction_losses + adv_losses

            test_epoch_loss += total_loss.item()
            test_epoch_reconstruction_loss += reconstruction_losses.item()
            test_epoch_prediction_loss += prediction_losses.item()
            test_epoch_adv_loss += adv_losses.item()
            test_epoch_ortho_loss += ortho_losses.item()

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
        test_ortho_loss.append(test_epoch_ortho_loss / len(target_test_loader))


    # AUROC (AUC-ROC)
    train_auroc = roc_auc_score(y_true_train, y_pred_train)
    test_auroc = roc_auc_score(y_true_test, y_pred_test)
    target_train_auroc = roc_auc_score(target_y_true_train, target_y_pred_train)
    target_test_auroc = roc_auc_score(target_y_true_test, target_y_pred_test)
    target_all_auroc = roc_auc_score(target_y_true_all, target_y_pred_all)

    train_aurocs.append(train_auroc)
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

    if test_epoch_loss < best_loss:
        best_loss = test_epoch_loss
        best_model_state_dict = model.state_dict()
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1


# Print progress
print(f"Epoch {epoch + 1}: "
      f"Train Loss: {train_total_loss[-1]:.5f}, "
      f"Test Loss: {test_total_loss[-1]:.5f}, "
      f"Train AUC: {train_aurocs[-1]:.5f}, "
      f"Test AUC: {test_aurocs[-1]:.5f}, "
      f"Target Train AUC: {target_train_aurocs[-1]:.5f}, "
      f"Target Test AUC: {target_test_aurocs[-1]:.5f}, "
      f"Target All AUC: {target_all_aurocs[-1]:.5f}, "
      f"Target test AP: {target_test_aps[-1]:.5f}")

torch.save(model.state_dict(), drug+'_model_parameter.pkl')