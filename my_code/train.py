if __name__ == '__main__':
    import os
    import sys
    import time
    import argparse
    import copy
    import configparser
    import numpy as np
    from tqdm import tqdm
    import matplotlib.pyplot as plt  # 新增

    # Torch Library.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    # Relative package.
    sys.path.append('../../my_code/')
    from my_code.utils.dataloader import DatasetLoader
    from my_code.utils.metrics import RegressionLoss, RegressionMetrics
    from my_code.utils.metrics import ClassificationLoss, ClassificationMetrics
    from my_code.utils.utils import compute_sampling_threshold, print_model_parameters
    from my_code.model.AGLSTAN import AGLSTAN

    # Testing print info.
    from icecream import ic
    from sklearn.metrics import f1_score  # 新增

    # 禁用 cuDNN
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # two city.
    # chi : Chicago
    # ny : New York
    CITY = 'chi'
    # CITY = 'ny'

    ############################InitSeed###################################
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ############################Arguments###################################
    # Read configure file
    config = configparser.ConfigParser()
    config.read(f'../data/{CITY}/config.conf')

    # set arguments
    args = argparse.ArgumentParser(description='arguments')

    # data
    args.add_argument('--adj_filename', default=config['data']['adj_filename'], type=str)
    args.add_argument('--node_features_filename', default=config['data']['node_features_filename'], type=str)
    args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
    args.add_argument('--window', default=config['data']['window'], type=int)
    args.add_argument('--horizon', default=config['data']['horizon'], type=int)
    args.add_argument('--default_graph', default=config['data']['default_graph'], type=str)
    # model
    args.add_argument('--model_name', default=config['model']['model_name'], type=str)
    args.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
    args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
    args.add_argument('--cheb_k', default=config['model']['cheb_k'], type=int)
    args.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
    args.add_argument('--filter_type', default=config['model']['filter_type'], type=str)
    args.add_argument('--activation_func', default=config['model']['activation_func'], type=str)
    args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
    args.add_argument('--filter_size', default=config['model']['filter_size'], type=int)
    # train
    args.add_argument('--epoch', default=config['train']['epoch'], type=int)
    args.add_argument('--lr', default=config['train']['lr'], type=float)
    args.add_argument('--factor', default=config['train']['factor'], type=float)
    args.add_argument('--patience', default=config['train']['patience'], type=float)
    args.add_argument('--early_stop', default=config['train']['early_stop'], type=int)
    args.add_argument('--train_loss_filename', default=config['train']['train_loss_filename'], type=str)
    args.add_argument('--val_loss_filename', default=config['train']['val_loss_filename'], type=str)
    args.add_argument('--binary', default=config['train']['binary'], type=str)
    args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
    args.add_argument('--loss_function', default=config['train']['loss_function'], type=str)
    args.add_argument('--cf_decay_steps', default=config['train']['cf_decay_steps'], type=int)
    args.add_argument('--teacher_forcing', default=config['train']['teacher_forcing'], type=str)
    # other
    args.add_argument('--device', default=config['other']['device'], type=str)
    args.add_argument('--data_path', default=config['other']['data_path'], type=str)
    args.add_argument('--res_path', default=config['other']['res_path'], type=str)
    args = args.parse_args()

    ##############################DataLoder#################################
    loader = DatasetLoader(args)
    data, adj, scaler, pos_weight, threshold = loader.get_dataset()
    train_loader, val_loader, test_loaders = data
    # with pre-defined or not
    if args.default_graph == 'true':
        adj = None

    ##############################ModelLoder#################################
    device = torch.device(args.device)
    model = AGLSTAN(args)
    criterion = ClassificationLoss(pos_weight=pos_weight, device=device)#新增语句
    model.to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    print("#" * 40 + "Model Info" + "#" * 40)
    print_model_parameters(model, only_num=False)

    ##############################Optm&Loss#################################
    # optimizer = optim.Adam(model.parameters(), lr=args.lr) #修改前
    optimizer = optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=args.lr)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    lr_decay_steps = [15, 25, 35, 55, 75]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                     milestones=lr_decay_steps,
                                                     gamma=0.5)
    if args.binary == 'false':
        criterion = RegressionLoss(scaler, mask_value=args.mask_value, loss_type=args.loss_function)
        metrics = RegressionMetrics(scaler, mask_value=args.mask_value)
    else:
        # criterion = ClassificationLoss(pos_weight=pos_weight.to(device), loss_type=args.loss_function, lambda_value=0.1,
        #                                device=device) #修改前
        criterion = ClassificationLoss(pos_weight=pos_weight.to(device),device=device)
        # criterion = ClassificationLoss(loss_type=args.loss_function, lambda_value=0.5, device=device)
        metrics = ClassificationMetrics(threshold)

    ##############################Training#################################
    print("#" * 40 + "Training" + "#" * 40)
    init_time = time.time()

    train_loss_list = []
    val_loss_list = []
    not_imporved = 0

    # 初始化指标列表
    train_bce_list = []
    train_micro_f1_list = []
    train_macro_f1_list = []
    val_bce_list = []
    val_micro_f1_list = []
    val_macro_f1_list = []

    for epoch in range(args.epoch):
        # 训练模式
        model.train()
        train_loss = 0
        all_train_preds = []
        all_train_labels = []

        for idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            if args.teacher_forcing == 'true':
                global_step = epoch * len(train_loader) + idx
                teacher_forcing_ratio = compute_sampling_threshold(global_step, args.cf_decay_steps)
            else:
                teacher_forcing_ratio = 1.

            preds = model(inputs, labels, teacher_forcing_ratio)  # teacher forcing
            # print(preds.size())
            # print(labels.size())
            loss = criterion(preds, labels)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            # 记录训练集的预测和标签
            all_train_preds.append(preds.detach().cpu())
            all_train_labels.append(labels.detach().cpu())

        # 计算平均训练损失
        train_loss = train_loss / len(train_loader)
        train_loss_list.append(train_loss)

        # 计算训练集的 F1 分数
        all_train_preds = torch.cat(all_train_preds, dim=0).numpy()
        all_train_labels = torch.cat(all_train_labels, dim=0).numpy()

        # 确保预测值通过 sigmoid 激活并二值化
        all_train_preds = (torch.sigmoid(torch.tensor(all_train_preds)) > 0.5).numpy().astype(int)
        all_train_labels = all_train_labels.astype(int)

        # 调试信息
        # print('Epoch {} Training:'.format(epoch))
        # print('y_true shape:', all_train_labels.shape)
        # print('y_pred shape:', all_train_preds.shape)
        # print('y_true unique values:', np.unique(all_train_labels))
        # print('y_pred unique values:', np.unique(all_train_preds))

        # 确保一维数组
        if all_train_labels.ndim > 1:
            all_train_labels = all_train_labels.flatten()
        if all_train_preds.ndim > 1:
            all_train_preds = all_train_preds.flatten()

        # 再次检查唯一值
        # print('After flattening:')
        # print('y_true unique values:', np.unique(all_train_labels))
        # print('y_pred unique values:', np.unique(all_train_preds))

        # 计算 F1 分数
        try:
            train_micro_f1 = f1_score(all_train_labels, all_train_preds, average='micro')
            train_macro_f1 = f1_score(all_train_labels, all_train_preds, average='macro')
        except ValueError as e:
            print("Error calculating F1 Score:", e)
            train_micro_f1 = 0.0
            train_macro_f1 = 0.0

        # 记录训练指标
        train_bce_list.append(train_loss)
        train_micro_f1_list.append(train_micro_f1)
        train_macro_f1_list.append(train_macro_f1)

        # 验证模式
        model.eval()
        val_loss = 0
        val_metrics = []
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs, labels, teacher_forcing_ratio=0.)  # not teacher forcing
                loss = criterion(preds, labels)
                val_loss += loss.item()
                val_metrics.append(metrics(preds, labels))

                # 记录验证集的预测和标签
                all_val_preds.append(preds.detach().cpu())
                all_val_labels.append(labels.detach().cpu())

        # 计算平均验证损失
        val_loss = val_loss / len(val_loader)
        val_loss_list.append(val_loss)

        # 计算验证集的 F1 分数
        all_val_preds = torch.cat(all_val_preds, dim=0).numpy()
        all_val_labels = torch.cat(all_val_labels, dim=0).numpy()

        all_val_preds = (torch.sigmoid(torch.tensor(all_val_preds)) > 0.5).numpy().astype(int)
        all_val_labels = all_val_labels.astype(int)

        # 调试信息
        # print('Epoch {} Validation:'.format(epoch))
        # print('y_true shape:', all_val_labels.shape)
        # print('y_pred shape:', all_val_preds.shape)
        # print('y_true unique values:', np.unique(all_val_labels))
        # print('y_pred unique values:', np.unique(all_val_preds))

        # 确保一维数组
        if all_val_labels.ndim > 1:
            all_val_labels = all_val_labels.flatten()
        if all_val_preds.ndim > 1:
            all_val_preds = all_val_preds.flatten()

        # 再次检查唯一值
        # print('After flattening:')
        # print('y_true unique values:', np.unique(all_val_labels))
        # print('y_pred unique values:', np.unique(all_val_preds))

        # 计算 F1 分数
        try:
            val_micro_f1 = f1_score(all_val_labels, all_val_preds, average='micro')
            val_macro_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
        except ValueError as e:
            print("Error calculating F1 Score:", e)
            val_micro_f1 = 0.0
            val_macro_f1 = 0.0

        # 记录验证指标
        val_bce_list.append(val_loss)
        val_micro_f1_list.append(val_micro_f1)
        val_macro_f1_list.append(val_macro_f1)

        ic(optimizer.param_groups[0]['lr'])

        # 调整学习率
        scheduler.step()

        # 打印训练和验证信息
        if args.binary == 'false':
            # 回归任务的打印信息（保持不变）
            val_rmse = torch.mean(torch.stack(list(map(lambda x: x[0], val_metrics))))
            val_mape = torch.mean(torch.stack(list(map(lambda x: x[1], val_metrics))))

            print(
                "epoch : {}, train_{}: {:.4f}, val_{}: {:.4f}, val_rmse: {:.4f}, val_mape: {:.4f}, duration: {:.2f}s".format(
                    epoch,
                    args.loss_function,
                    train_loss,
                    args.loss_function,
                    val_loss,
                    val_rmse.cpu().item(),
                    val_mape.cpu().item(),
                    time.time() - init_time
                ))
        else:
            print(
                "epoch : {}, train_bce: {:.4f}, train_micro_f1: {:.4f}, train_macro_f1: {:.4f}, val_bce: {:.4f}, val_micro_f1: {:.4f}, val_macro_f1: {:.4f}, duration: {:.2f}s".format(
                    epoch,
                    train_loss,
                    train_micro_f1,
                    train_macro_f1,
                    val_loss,
                    val_micro_f1,
                    val_macro_f1,
                    time.time() - init_time
                ))

        # 提前停止和保存最佳模型
        if min(val_loss_list) >= val_loss_list[-1]:
            state = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': args
            }
            torch.save(state, os.path.join(args.res_path, args.model_name))
            not_imporved = 0
            print("The Best Model has been saved!")

        elif not_imporved == args.early_stop:
            print("Validation performance didn\'t improve for {} epochs. Training stops.".format(args.early_stop))
            print("The best val_loss : {:.4f}".format(min(val_loss_list)))
            break

        else:
            not_imporved += 1

    ##############################Testing#################################
    print("#" * 40 + "Testing" + "#" * 40)
    best_model = torch.load(os.path.join(args.res_path, args.model_name))
    state_dict = best_model['state_dict']
    args = best_model['config']
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    test_dataset_name = ['Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month, test_loader in enumerate(test_loaders):
        test_loss = 0
        test_metrics = []
        all_test_preds = []
        all_test_labels = []

        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(test_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                preds = model(inputs, labels, teacher_forcing_ratio=0.)  # not teacher forcing

                test_loss += criterion(preds, labels).item()
                test_metrics.append(metrics(preds, labels))

                # 记录测试集的预测和标签
                all_test_preds.append(preds.detach().cpu())
                all_test_labels.append(labels.detach().cpu())

        test_loss = test_loss / len(test_loader)

        # 计算测试集的 F1 分数
        all_test_preds = torch.cat(all_test_preds, dim=0).numpy()
        all_test_labels = torch.cat(all_test_labels, dim=0).numpy()

        all_test_preds = (torch.sigmoid(torch.tensor(all_test_preds)) > 0.5).numpy().astype(int)
        all_test_labels = all_test_labels.astype(int)

        # 确保一维数组
        if all_test_labels.ndim > 1:
            all_test_labels = all_test_labels.flatten()
        if all_test_preds.ndim > 1:
            all_test_preds = all_test_preds.flatten()

        # 计算 F1 分数
        try:
            test_micro_f1 = f1_score(all_test_labels, all_test_preds, average='micro')
            test_macro_f1 = f1_score(all_test_labels, all_test_preds, average='macro')
        except ValueError as e:
            print("Error calculating Test F1 Score:", e)
            test_micro_f1 = 0.0
            test_macro_f1 = 0.0

        print("Test on {} dataset".format(test_dataset_name[month]))
        if args.binary == 'false':
            test_rmse = torch.mean(torch.stack(list(map(lambda x: x[0], test_metrics))))
            test_mape = torch.mean(torch.stack(list(map(lambda x: x[1], test_metrics))))

            print("test_{}: {:.4f}, test_rmse: {:.4f}, test_mape: {:.4f}".format(
                args.loss_function,
                test_loss,
                test_rmse.cpu().item(),
                test_mape.cpu().item()
            ))
        else:
            print("test_{}: {:.4f}, test_micro_f1: {:.4f}, test_macro_f1: {:.4f}".format(
                args.loss_function,
                test_loss,
                test_micro_f1,
                test_macro_f1
            ))

    ############################Save Results###############################
    # 保存训练和验证损失
    np.save(os.path.join(args.res_path, args.train_loss_filename),
            np.array([each for each in train_loss_list]))
    np.save(os.path.join(args.res_path, args.val_loss_filename),
            np.array([each for each in val_loss_list]))

    # 保存训练好的节点嵌入
    params = dict()
    for name, param in list(model.named_parameters()):
        params[name] = param
    node_embeddings = params['node_embeddings']
    supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
    support_set = [torch.eye(args.num_nodes).to(supports.device), supports]
    # supports = torch.stack(support_set, dim=0)
    np.save(os.path.join(args.res_path, 'adj.npy'), supports.cpu().detach().numpy())

    # 绘制指标变化图
    epochs = range(1, len(train_bce_list) + 1)

    plt.figure(figsize=(12, 12))

    # 绘制 BCE 损失
    plt.subplot(3, 1, 1)
    plt.plot(epochs, train_bce_list, 'b-', label='Train BCE')
    plt.plot(epochs, val_bce_list, 'r-', label='Validation BCE')
    plt.title('BCE Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.legend()

    # 绘制 Micro F1 分数
    plt.subplot(3, 1, 2)
    plt.plot(epochs, train_micro_f1_list, 'g-', label='Train Micro F1')
    plt.plot(epochs, val_micro_f1_list, 'm-', label='Validation Micro F1')
    plt.title('Micro F1 Score over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Micro F1 Score')
    plt.legend()

    # 绘制 Macro F1 分数
    plt.subplot(3, 1, 3)
    plt.plot(epochs, train_macro_f1_list, 'c-', label='Train Macro F1')
    plt.plot(epochs, val_macro_f1_list, 'y-', label='Validation Macro F1')
    plt.title('Macro F1 Score over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Macro F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.res_path, 'metrics_over_epochs.png'))
    plt.show()

    data_to_save = np.vstack((epochs, train_bce_list, val_bce_list, train_micro_f1_list, val_micro_f1_list,
                              train_macro_f1_list, val_macro_f1_list)).T
    np.savetxt(os.path.join(args.res_path,'data.csv'), data_to_save, delimiter=',',
               header='epochs,train_bce,val_bce,train_micro_f1,val_micro_f1,train_macro_f1,val_macro_f1')

    import torch

    # 在训练完成后，或者最佳模型保存时，保存模型
    save_path = r"D:\111111111111111111ArealCode\AGL-STAN-main\res\chi\zuizhong.pth"
    torch.save(model.state_dict(), save_path)

    print(f"模型已保存到 {save_path}")
