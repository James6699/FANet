import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import torch.autograd
import torch.nn.functional as F
from torchvision import transforms
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import time
from PIL import Image
import yaml
from models.FANet_tan_level1 import FANet
from data.dataset import SOD360Dataset
# from models.FANet_tan_level1 import FANet
# from data.dataset_tan_level1 import SOD360Dataset
from utils import AverageMeter

tensor2img = transforms.ToPILImage()
def get_1x_lr_params(model):
    b = []
    b.append(model.basenet.conv1)
    b.append(model.basenet.bn1)
    b.append(model.basenet.layer1)
    b.append(model.basenet.layer2)
    b.append(model.basenet.layer3)
    # b.append(model.basenet.layer4)
    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    # k.requires_grad = False
                    yield k
                    # yield k


def get_10x_lr_params(model):
    b = []
    b.append(model.FPN.parameters())
    b.append(model.PFFusion.parameters())
    b.append(model.MLFCombination.parameters())


    for j in range(len(b)):
        for i in b[j]:
            yield i


###############################################################################
def adjust_learning_rate(cfg, optimizer, epoch, i_iter, dataset_lenth):
    lr = cfg['lr']*((1-float(epoch*dataset_lenth+i_iter)/(cfg['epochs']*dataset_lenth))**(cfg['power']))
    print('Epoch [{}] Learning rate: {}'.format(epoch, lr))
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * 10

def train(train_loader, model, criterion1, criterion2, optimizer, epoch, out_model_path, log_dir_path, init_iter, cfg):
    epoch_nums = 0
    assert cfg['use_gpu']
    model.train()
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_1 = AverageMeter()
    writer = SummaryWriter(log_dir_path)

    for i_iter, batch in enumerate(train_loader):
        ttime = time.time()
        i_iter += init_iter
        adjust_learning_rate(cfg, optimizer, epoch, i_iter, len(train_loader))
        input_list, label_list, _, _ = batch


        equi_img = input_list[0].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
        # equi_img = F.interpolate(equi_img, scale_factor=0.5, mode='bilinear', align_corners=True)
        tan_0 = input_list[1].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
        tan_1 = input_list[2].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
        tan_2 = input_list[3].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
        tan_3 = input_list[4].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
        tan_4 = input_list[5].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
        tan_5 = input_list[6].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
        tan_6 = input_list[7].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
        tan_7 = input_list[8].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
        tan_8 = input_list[9].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
        tan_9 = input_list[10].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
        tan_10 = input_list[11].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
        tan_11 = input_list[12].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
        tan_12 = input_list[13].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
        tan_13 = input_list[14].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
        tan_14 = input_list[15].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
        tan_15 = input_list[16].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
        tan_16 = input_list[17].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
        tan_17 = input_list[18].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
        tan_18 = input_list[19].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
        tan_19 = input_list[20].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))

        
        equi_label = label_list[0].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))

        # pred_list = model([equi_img,tan_0,tan_1,tan_2,tan_3,tan_4,tan_5,tan_6,tan_7,tan_8,tan_9,
        #                    tan_10,tan_11,tan_12,tan_13,tan_14,tan_15,tan_16,tan_17,tan_18,tan_19,
        #                    tan_20,tan_21,tan_22,tan_23,tan_24,tan_25,tan_26,tan_27,tan_28,tan_29,
        #                    tan_30,tan_31,tan_32,tan_33,tan_34,tan_35,tan_36,tan_37,tan_38,tan_39,
        #                    tan_40,tan_41,tan_42,tan_43,tan_44,tan_45,tan_46,tan_47,tan_48,tan_49,
        #                    tan_50,tan_51,tan_52,tan_53,tan_54,tan_55,tan_56,tan_57,tan_58,tan_59,
        #                    tan_60,tan_61,tan_62,tan_63,tan_64,tan_65,tan_66,tan_67,tan_68,tan_69,
        #                    tan_70,tan_71,tan_72,tan_73,tan_74,tan_75,tan_76,tan_77,tan_78,tan_79])
        pred_list = model([equi_img,tan_0,tan_1,tan_2,tan_3,tan_4,tan_5,tan_6,tan_7,tan_8,tan_9,
                           tan_10,tan_11,tan_12,tan_13,tan_14,tan_15,tan_16,tan_17,tan_18,tan_19])

        equi_pred = nn.functional.interpolate(pred_list[0], size=(cfg['equi_input_height'], cfg['equi_input_width']), mode='bilinear', align_corners=True)
        equi_l1pred = nn.functional.interpolate(pred_list[1], size=(cfg['equi_input_height'], cfg['equi_input_width']), mode='bilinear', align_corners=True)
        equi_l2pred = nn.functional.interpolate(pred_list[2], size=(cfg['equi_input_height'], cfg['equi_input_width']), mode='bilinear', align_corners=True)
        equi_l3pred = nn.functional.interpolate(pred_list[3], size=(cfg['equi_input_height'], cfg['equi_input_width']), mode='bilinear', align_corners=True)
        # equi_l4pred = nn.functional.interpolate(torch.sigmoid(pred_list[4]), size=(cfg['equi_input_height'], cfg['equi_input_width']), mode='bilinear', align_corners=True)

        # equi_l1pred = equi_l1pred.squeeze()
        # equi_l2pred = equi_l2pred.squeeze()
        # equi_l3pred = equi_l3pred.squeeze()
        # equi_l4pred = equi_l4pred.squeeze()
        # equi_l1pred = np.array(tensor2img(equi_l1pred.data.squeeze().cpu())).astype(np.uint8)
        # equi_l1pred = Image.fromarray(equi_l1pred)
        # equi_l2pred = np.array(tensor2img(equi_l2pred.data.squeeze().cpu())).astype(np.uint8)
        # equi_l2pred = Image.fromarray(equi_l2pred)
        # equi_l3pred = np.array(tensor2img(equi_l3pred.data.squeeze().cpu())).astype(np.uint8)
        # equi_l3pred = Image.fromarray(equi_l3pred)
        # equi_l4pred = np.array(tensor2img(equi_l4pred.data.squeeze().cpu())).astype(np.uint8)
        # equi_l4pred = Image.fromarray(equi_l4pred)


        equi_loss1 = criterion1(equi_pred, equi_label)
        equi_loss2 = criterion2(torch.sigmoid(equi_pred), equi_label)
        equi_loss = equi_loss2 + equi_loss1

        equi_l1loss1 = criterion1(equi_l1pred, equi_label)
        equi_l1loss2 = criterion2(torch.sigmoid(equi_l1pred), equi_label)
        equi_l1loss = equi_l1loss1 + equi_l1loss2

        equi_l2loss1 = criterion1(equi_l2pred, equi_label)
        equi_l2loss2 = criterion2(torch.sigmoid(equi_l2pred), equi_label)
        equi_l2loss = equi_l2loss1 + equi_l2loss2

        equi_l3loss1 = criterion1(equi_l3pred, equi_label)
        equi_l3loss2 = criterion2(torch.sigmoid(equi_l3pred), equi_label)
        equi_l3loss = equi_l3loss1 + equi_l3loss2

        # equi_l4loss1 = criterion1(equi_l4pred, equi_label)
        # equi_l4loss2 = criterion2(torch.sigmoid(equi_l4pred), equi_label)
        # equi_l4loss = equi_l4loss1 + equi_l4loss2

        loss = equi_loss + equi_l1loss + equi_l2loss + equi_l3loss
        # loss = equi_loss  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update((time.time() - ttime) / cfg['summary_freq'])
        losses.update(loss.data.item(), cfg['batch_size'])
        losses_1.update(equi_loss.data.item(), cfg['batch_size'])
        if i_iter % cfg['summary_freq'] == 0:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    epoch, cfg['epochs'], i_iter, len(train_loader), batch_time=batch_time,
                    loss=losses),'L:',losses_1.avg)
        writer.add_scalar('Train/Batch_Loss', losses.val, i_iter+epoch*len(train_loader))
        writer.add_scalar('Train/Epoch_Loss', losses.avg, epoch)
        writer.add_scalar('Train/L_Loss', losses_1.avg, epoch)
        # writer.add_image("prel1", equi_l1pred,1)
        # writer.add_image("prel2", equi_l2pred,2)
        # writer.add_image("prel3", equi_l3pred,3)
        # writer.add_image("prel4", equi_l4pred,4)

        if i_iter % len(train_loader) == (len(train_loader)-1):
            # if epoch>=40 and epoch % 4 == 0:
            if epoch >= 10 and epoch % 5 == 0:
                print( 'saving checkpoint: ', os.path.join(out_model_path, '{0}_{1:02}.pth'.format(cfg['model_name'], epoch)))
                torch.save(model.state_dict(), os.path.join(out_model_path,
                        '{0}_{1:02}.pth'.format(cfg['model_name'], epoch)))


def main():
    with open('./config.yaml') as f:
        config = yaml.safe_load(f)
    for key in config.keys():
        print("\t{} : {}".format(key, config[key]))
    print('---------------------------------------------------------------' + '\n')

    log_dir_path = os.path.join(config['logs'], "{0}/{1}".format(config['model_name'], 'train'))
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)

    out_model_path = os.path.join(config['checkpoints'], "{0}".format(config['model_name']))
    if not os.path.exists(out_model_path):
        os.makedirs(out_model_path)

    train_dataset = SOD360Dataset(config['train_data'], config['train_list'], (config['equi_input_width'], config['equi_input_height']),config['padding_size'])
    train_loader = data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['processes'], pin_memory=True)

    model = FANet(num_classes=config['num_classes'])

    print(torch.cuda.is_available())
    if config['use_gpu']:
        # model.half()
        model = model.to(torch.device('cuda', config['device_id']))

    model_num = []
    init_model_num=0
    if model_num != []:
        model.load_state_dict(torch.load(os.path.join(out_model_path,
                        '{0}_{1}.pth'.format(config['model_name'],max(model_num)))))
        init_model_num = max(model_num)
        print(init_model_num)

    if config['use_gpu']:
        criterion1 = nn.BCEWithLogitsLoss().to(torch.device('cuda', config['device_id']))
        criterion2 = nn.L1Loss().to(torch.device('cuda', config['device_id']))
    else:
        criterion1 = nn.BCEWithLogitsLoss()
        criterion2 = nn.L1Loss()

    optimizer = torch.optim.SGD([{'params': get_1x_lr_params(model),'lr': config['lr']},{'params': get_10x_lr_params(model),'lr': config['lr'] * 10}],
                                lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])

    for epoch in range(config['epochs']):
        train(train_loader, model, criterion1, criterion2, optimizer, epoch, out_model_path, log_dir_path, init_model_num, config)


if __name__ == '__main__':
    main()