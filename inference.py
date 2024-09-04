import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import torch.autograd
from torchvision import transforms
from PIL import Image
import time
import yaml
from torch.utils.tensorboard import SummaryWriter
# from models.FANet_tan_level1 import FANet
# from data.dataset_tan_level1 import SOD360Dataset
from models.FANet_tan_level1 import FANet
from data.dataset import SOD360Dataset
from utils import AverageMeter
from evaluateSOD.main import evalateSOD


tensor2img = transforms.ToPILImage()
def inference(test_loader, model, cfg, result_path):
    assert cfg['use_gpu']
    criterion1 = nn.BCEWithLogitsLoss().to(torch.device('cuda',0))
    criterion2 = nn.L1Loss().to(torch.device('cuda',0))
    with torch.no_grad():
        for i_iter, batch in enumerate(test_loader):
            input_list, label_list, name, img_size = batch
            equi_img = input_list[0].type(torch.FloatTensor).to(torch.device('cuda', cfg['device_id']))
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

            pred_list = model([equi_img,tan_0,tan_1,tan_2,tan_3,tan_4,tan_5,tan_6,tan_7,tan_8,tan_9,
                tan_10,tan_11,tan_12,tan_13,tan_14,tan_15,tan_16,tan_17,tan_18,tan_19])
            # equi_pred = nn.functional.interpolate(pred_list[0], size=(cfg['equi_input_height'], cfg['equi_input_width']), mode='bilinear', align_corners=True)
            # equi_loss1 = criterion1(equi_pred, equi_label)
            # equi_loss2 = criterion2(torch.sigmoid(equi_pred), equi_label)
            # equi_loss = equi_loss1 + equi_loss2
            # print(equi_loss)
            equi_pred = nn.functional.interpolate(torch.sigmoid(pred_list[0]), size=(img_size[1], img_size[0]), mode='bilinear', align_corners=True)
            # equi_pred = nn.functional.interpolate(torch.sigmoid(pred_list[0]), size=(512,1024), mode='bilinear', align_corners=True)
            equi_pred = np.array(tensor2img(equi_pred.data.squeeze(0).cpu())).astype(np.uint8)
            equi_pred = Image.fromarray(equi_pred)

            # equi_pred_mid = nn.functional.interpolate(torch.sigmoid(pred_list[1]), size=(img_size[1], img_size[0]), mode='bilinear', align_corners=True)
            # equi_pred_mid = np.array(tensor2img(equi_pred_mid.data.squeeze(0).cpu())).astype(np.uint8)
            # equi_pred_mid = Image.fromarray(equi_pred_mid)

            # tan_pre = nn.functional.interpolate(torch.sigmoid(pred_list[2]), size=(img_size[1], img_size[0]), mode='bilinear', align_corners=True)
            # tan_pre = np.array(tensor2img(tan_pre.data.squeeze(0).cpu())).astype(np.uint8)
            # tan_pre = Image.fromarray(tan_pre)

            if not os.path.exists(result_path):
                os.makedirs(result_path)
            equi_pred.save(os.path.join(result_path, name[0] + '.png'))
            # equi_pred_mid.save(os.path.join(result_path, name[0]+'_erp_mid' + '.png'))
            # tan_pre.save(os.path.join(result_path, name[0]+'_tan' + '.png'))



def main():
    with open('./config.yaml') as f:
        config = yaml.safe_load(f)
    for key in config.keys():
        print("\t{} : {}".format(key, config[key]))
    print('---------------------------------------------------------------' + '\n')

    result_path = os.path.join(config['result'], "{0}/{1}".format(config['testset_name'], config['model_name']))
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    load_model_path = os.path.join(config['checkpoints'], "{0}".format(config['model_name']))
    if not os.path.exists(load_model_path):
        os.makedirs(load_model_path)

    # model = FANet(num_classes=config['num_classes'],padding_size=config['padding_size'],blend=config['blend'])
    model = FANet(num_classes=config['num_classes'])
    if config['use_gpu']:
        # model.half()
        model = model.to(torch.device('cuda', config['device_id']))


    test_dataset = SOD360Dataset(config['test_data'], config['test_list'],
                                 (config['equi_input_width'], config['equi_input_height']), config['padding_size'],train=False)
    test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                  num_workers=config['processes'], pin_memory=True)

    print('Inferring the data on %d-th iteration model ' % (config['model_id']+1))
    ckpt_name = 'FANet_80.pth'
    # saved_state_dict = torch.load(os.path.join(load_model_path,'{0}_{1:02}.pth'.format(config['model_name'], config['model_id'])))
    saved_state_dict = torch.load(os.path.join(load_model_path,ckpt_name))
    model.load_state_dict(saved_state_dict)

    model.eval()

    inference(test_loader, model, config, result_path)
    gt_root = '/media/mmclxk/data/FANet-master/Dataset/F-360iSOD/gt_train'
    dataset = 'F-360iSOD'
    
    _ = evalateSOD(result_path, gt_root, dataset,ckpt_name)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end - start, 'seconds')