import torch
from Dataset.transform import UnNormalize
from utils.misc import AverageMeter
import time
import numpy as np
import torchvision
def train_one_epoch(train_loader, model, optimizer, criterion, lr_scheduler, epoch,  args, logger):
    batch_time = AverageMeter()  #一个batch中模型运行时间
    data_time = AverageMeter()  #一个batch中加载数据时间
    loss = AverageMeter()
    mae = AverageMeter()
    acc = AverageMeter()
    model.train()
    batch_start_time = time.time()

    for idx, (imgs, labels) in enumerate(train_loader):
        if epoch <= 0 and idx <= 10:
            ori_imgs=UnNormalize(mean =[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(imgs)
            torchvision.utils.save_image(ori_imgs[0],'images/epoch_{:}image_{:}.jpg'.format(epoch, idx),normalize=True)
        imgs = imgs.cuda()
        labels = labels.cuda()
        bs = labels.shape[0]
        data_time.update(time.time() - batch_start_time)
        optimizer.zero_grad()
        output = model(imgs)
        loss_train = criterion(100*output, 100*labels) #这个batch的平均损失, 每个样本在每个AU上的平均均方损失
        loss_train.backward()
        optimizer.step()
        lr_scheduler.step_update((epoch * len(train_loader) + idx))
        lr = optimizer.param_groups[0]['lr']

        loss.update(loss_train.item(), bs)
        batch_time.update(time.time() - batch_start_time)
        batch_start_time = time.time()

        mae_array = torch.mean(torch.abs(labels-output), dim=1)  #batchsize个样本的每个AU的平均绝对值损失
        mae_value = torch.mean(mae_array)   #每个样本在每个AU上的平均绝对值损失
        
        max_mae_array = torch.max(torch.abs(labels-output), dim=1)[0] #batchsize个样本的每个AU的最大绝对值损失
        predict_true_array = torch.where(max_mae_array < 0.08, 1, 0) #根据阈值（0.08）将绝对值损失转换为预测的真值，存储在 pred_mae_array 中
        
        #predict_true_array = torch.where(mae_array < 0.08, 1, 0) #根据阈值（0.08）将绝对值损失转换为预测的真值，存储在 predict_true_array 中
        acc_value = torch.sum(predict_true_array) / bs #计算准确度，即在每个AU上绝对值损失低于阈值的比例

        mae.update(mae_value, bs)#更新均方误差的平均值。
        acc.update(acc_value, bs)#更新准确度的平均值
        
        # # global_steps = writer_dict['train_global_steps']
        # #import ipdb;ipdb.set_trace()
        if idx % args.print_fq == 0 or idx + 1 == len(train_loader): #满足条件输出日志信息
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                    'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                    'Speed {speed:.1f} samples/s\t' \
                    'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                    'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                    'LR {lr:.5f}\t' \
                    'MAE {mae.val:.5f} ({mae.avg:.5f})\t' \
                    'ACC {acc.val:.5f} ({acc.avg:.5f})\t' .format(
                epoch, idx, len (train_loader), batch_time=batch_time,
                speed=bs / batch_time.val,
                data_time=data_time, loss=loss, lr=lr, mae=mae,acc=acc)
            logger.info (msg)

        # writer = writer_dict['writer']
        # writer.add_scalar('train_loss', loss.val, global_steps)
        # writer.add_scalar('train_acc', mae.val, global_steps)
        # writer_dict['train_global_steps'] = global_steps + 1

    return acc.avg, mae.avg

def evalutate(val_loader, model, criterion, epoch, args, logger):
    batch_time = AverageMeter()  #一个batch中模型运行时间
    data_time = AverageMeter()  #一个batch中加载数据时间
    loss = AverageMeter()
    mae = AverageMeter()
    non_zero_mae = AverageMeter()
    acc = AverageMeter()
    pl = AverageMeter()
    pm = AverageMeter()

    Nmae_for_each_au = torch.zeros(24).cuda()
    #icc = AverageMeter()
    batch_start_time = time.time()
    print("evaluating...")
    model.eval()
    pred = []
    label = []
    with torch.no_grad():
        for idx, (imgs, labels) in enumerate(val_loader):
            if epoch == 0 and idx <= 10 :
                ori_imgs=UnNormalize(mean =[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(imgs)
                torchvision.utils.save_image(ori_imgs[0],'images/val_image_{:}.jpg'.format(idx),normalize=True)
            imgs = imgs.cuda()
            labels = labels.cuda()
            bs = labels.shape[0]
            output = model(imgs)
            pred.append(np.array(output.cpu()))
            label.append(np.array(labels.cpu()))
            loss_val = criterion(100*output, 100*labels)
            loss.update(loss_val.item(), bs)
            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()
            #icc_value = cal_ICC(output.cpu().numpy(),labels.cpu().numpy())
            
            non_zero_mask = labels != 0
            non_zero_output = output[non_zero_mask]
            non_zero_labels = labels[non_zero_mask]
            non_zero_mae_value = torch.mean(torch.abs(non_zero_labels-non_zero_output))
            non_zero_mae.update(non_zero_mae_value.item(), len(non_zero_output))
            
            
            mae_array = torch.mean(torch.abs(labels-output), dim=1)  #batchsize个样本的每个AU的平均绝对值损失
            mae_value = torch.mean(mae_array)   #每个样本在每个AU上的平均绝对值损失
            Nmae_for_each_au += torch.sum(torch.abs(labels-output),dim=0)
            #predict_true_array = torch.where(mae_array < 0.08, 1, 0)
            
            max_mae_array = torch.max(torch.abs(labels-output), dim=1)[0] #batchsize个样本的每个AU的最大绝对值损失
            predict_true_array = torch.where(max_mae_array < 0.08, 1, 0) #根据阈值（0.08）将绝对值损失转换为预测的真值，存储在 pred_mae_array 中
            
            
            acc_value = torch.sum(predict_true_array) / bs
            pred_less = output < labels
            pred_less[labels==0] = 0
            pred_less_num = np.sum(np.array(pred_less.cpu()))
            pred_more = output > labels
            pred_more[labels==0] = 0
            pred_more_num = np.sum(np.array(pred_more.cpu()))

            pl.update(pred_less_num / bs, bs)
            pm.update(pred_more_num / bs, bs)
            mae.update(mae_value, bs)
            acc.update(acc_value, bs)
#           icc.update(icc_value.mean(), bs)
            # global_steps = writer_dict['train_global_steps']
            if idx % args.print_fq == 0 or idx + 1 == len(val_loader):
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                      'MAE {mae.val:.5f} ({mae.avg:.5f})\t' \
                      'ACC {acc.val:.5f} ({acc.avg:.5f})\t'\
                       'Non_zero_mae {non_zero_mae.val:.5f} ({non_zero_mae.avg:.5f})\t' \
                       .format (
                    epoch, idx, len (val_loader), batch_time=batch_time,
                    speed=bs / batch_time.val,
                    data_time=data_time, loss=loss, mae=mae, acc=acc, non_zero_mae=non_zero_mae)
                logger.info (msg)
        print("mae for each au is :", Nmae_for_each_au / acc.count)


            # val_loss += loss_val
            # mae = torch.mean(torch.abs(labels-output), dim=1)
            # acc_array = torch.where(mae < 0.08, 1, 0)

            # labels = labels > 0
            # output = output > 0
            # true_positives += torch.sum(labels & output, dim=0).cuda()
            # predicted_positives += torch.sum(output, dim=0).cuda()
            # actual_positives += torch.sum(labels, dim=0).cuda()
        # precision = true_positives / (predicted_positives + 1e-7)
        # recall = true_positives / (actual_positives + 1e-7)
        # f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        # average_f1 = f1.mean()
        # acc = right_preds / num_data
        # val_loss = val_loss / len(val_loader)
        # print("epochs: %d, val_loss:  %.5f, average_acc: %.5f"%(epoch, val_loss, acc))
        # print("f1:", f1)
        #print("average acc: ", torch.mean(acc))


        pred = np.concatenate(pred)
        label = np.concatenate(label)
        return loss.avg, acc.avg, mae.avg, pl.sum, pm.sum, pred, label, non_zero_mae.avg

def cal_ICC(predict, label):
    batch_size,num_au=label.shape[0],label.shape[1]
    iccs=[]
    for i in range(num_au): #对每一个AU标签计算对应的ICC值
        curlabel = label[:,i]
        curpre = predict[:,i]
        icc_data = np.stack((curlabel,curpre),axis=1)
        icc_value = icc(icc_data,icc_type='icc3')
        iccs.append(icc_value)
    iccs = np.array(iccs)
    return iccs

def icc(data, icc_type='icc2'):
    ''' Calculate intraclass correlation coefficient for data within
        Brain_Data class
    ICC Formulas are based on:
    Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in
    assessing rater reliability. Psychological bulletin, 86(2), 420.
    icc1:  x_ij = mu + beta_j + w_ij
    icc2/3:  x_ij = mu + alpha_i + beta_j + (ab)_ij + epsilon_ij
    Code modifed from nipype algorithms.icc
    https://github.com/nipy/nipype/blob/master/nipype/algorithms/icc.py
    Args:
        icc_type: type of icc to calculate (icc: voxel random effect,
                icc2: voxel and column random effect, icc3: voxel and
                column fixed effect)
    Returns:
        ICC: (np.array) intraclass correlation coefficient
    '''

    Y = data
    [n, k] = Y.shape

    # Degrees of Freedom
    dfc = k - 1
    dfe = (n - 1) * (k-1)
    dfr = n - 1

    # Sum Square Total
    mean_Y = np.mean(Y)
    SST = ((Y - mean_Y) ** 2).sum()

    # create the design matrix for the different levels
    x = np.kron(np.eye(k), np.ones((n, 1)))  # sessions k = 2 n =512 (1024,2)
    x0 = np.tile(np.eye(n), (k, 1))  # subjects  (1024,512)
    X = np.hstack([x, x0])  # (1024,514)
    # Sum Square Error
    predicted_Y = np.dot(np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))),
                                X.T), Y.flatten('F'))   #(1024,)
    residuals = Y.flatten('F') - predicted_Y
    SSE = (residuals ** 2).sum()

    MSE = SSE / dfe

    # Sum square column effect - between colums
    SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * n
    MSC = SSC / dfc / n

    # Sum Square subject effect - between rows/subjects
    SSR = SST - SSC - SSE
    MSR = SSR / dfr

    if icc_type == 'icc1':
        # ICC(2,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error +
        # k*(mean square columns - mean square error)/n)
        # ICC = (MSR - MSRW) / (MSR + (k-1) * MSRW)
        NotImplementedError("This method isn't implemented yet.")

    elif icc_type == 'icc2':
        # ICC(2,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error +
        # k*(mean square columns - mean square error)/n)
        ICC = (MSR - MSE) / (MSR + (k-1) * MSE + k * (MSC - MSE) / n)

    elif icc_type == 'icc3':
        # ICC(3,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error)
        ICC = (MSR - MSE) / (MSR + (k-1) * MSE + 1e-10)

    return ICC
