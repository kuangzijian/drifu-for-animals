import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import random
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from lib.options import BaseOptions
from lib.train_util import *
from lib.data import *
from lib.model import *

# get options
opt = BaseOptions().parse()

def train(opt):
    # set cuda
    cuda = torch.device('cuda:%d' % opt.gpu_id)
    writer = SummaryWriter()
    writer_validation = SummaryWriter()
    train_dataset = TrainDataset(opt, phase='train')
    test_dataset = TrainDataset(opt, phase='test')

    projection_mode = train_dataset.projection_mode

    # create data loader
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=opt.batch_size, shuffle=not opt.serial_batches,
                                   pin_memory=opt.pin_memory)

    print('train data size: ', len(train_data_loader))

    # NOTE: batch size should be 1 and use all the points for evaluation
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=1, shuffle=False,
                                  pin_memory=opt.pin_memory)
    print('test data size: ', len(test_data_loader))

    # create shape net camera net
    netG = HGPIFuNet(opt, projection_mode).to(device=cuda)
    netCam = CameraEncoder(opt).to(device=cuda)
    netC = ResBlkPIFuNet(opt).to(device=cuda)

    optimizerG = torch.optim.RMSprop(netG.parameters(), lr=opt.learning_rate, momentum=0, weight_decay=0)
    optimizerCam = torch.optim.Adam(netCam.parameters(), lr=opt.learning_rateCam, betas=(0.5, 0.999))
    optimizerC = torch.optim.Adam(netC.parameters(), lr=opt.learning_rate)

    lr = opt.learning_rate
    lrCam = opt.learning_rate
    print('Using Network: ', netG.name)
    print('Using Network: ', netCam.name)
    print('Using Network: ', netC.name)

    def set_train():
        netG.train()
        netCam.train()
        netC.train()

    def set_eval():
        netG.eval()
        netCam.eval()
        netC.eval()

    if opt.continue_train:
        resume_path = '%s/%s/netGandCam_latest' % (opt.checkpoints_path, opt.name)
        checkpoint = torch.load(resume_path)
        print('Resuming from ', resume_path)
        netG.load_state_dict(checkpoint['netG'])
        netCam.load_state_dict(checkpoint['netCam'])
        netC.load_state_dict(checkpoint['netC'])
        resume_epoch = checkpoint['epoch']

    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

    opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
    with open(opt_log, 'w') as outfile:
        outfile.write(json.dumps(vars(opt), indent=2))

    # training
    start_epoch = 0 if not opt.continue_train else resume_epoch
    for epoch in range(start_epoch, opt.num_epoch):
        epoch_start_time = time.time()

        set_train()
        iter_data_time = time.time()
        for train_idx, train_data in enumerate(train_data_loader):
            iter_start_time = time.time()

            # retrieve the data
            image_tensor = train_data['img'].to(device=cuda)
            calib_tensor = train_data['calib'].to(device=cuda)
            sample_tensor = train_data['samples'].to(device=cuda)
            camera_tensor = train_data['camera'].to(device=cuda)
            color_sample_tensor = train_data['color_samples'].to(device=cuda)

            image_tensor, calib_tensor, __ = reshape_multiview_tensors(image_tensor, calib_tensor, None)

            if opt.num_views > 1:
                sample_tensor = reshape_sample_tensor(sample_tensor, opt.num_views)
                color_sample_tensor = reshape_sample_tensor(color_sample_tensor, opt.num_views)

            label_tensor = train_data['labels'].to(device=cuda)
            rgb_tensor = train_data['rgbs'].to(device=cuda)

            res, error = netG.forward(image_tensor, sample_tensor, calib_tensor, labels=label_tensor)
            resCam, errorCam = netCam.forward(image_tensor, camera_tensor)
            with torch.no_grad():
                netG.filter(image_tensor)
            resC, errorC = netC.forward(image_tensor, netG.get_im_feat(), color_sample_tensor, calib_tensor, labels=rgb_tensor)

            writer.add_scalar("LossG/train", error, epoch)
            optimizerG.zero_grad()
            error.backward()
            optimizerG.step()

            writer.add_scalar("LossCam/train", errorCam, epoch)
            optimizerCam.zero_grad()
            errorCam.backward()
            optimizerCam.step()

            writer.add_scalar("LossC/train", errorC, epoch)
            optimizerC.zero_grad()
            errorC.backward()
            optimizerC.step()

            iter_net_time = time.time()
            eta = ((iter_net_time - epoch_start_time) / (train_idx + 1)) * len(train_data_loader) - (
                    iter_net_time - epoch_start_time)

            if train_idx % opt.freq_plot == 0:
                print(
                    'Shape & Camera: {0} | Epoch: {1} | {2}/{3} | Err: {4:.06f} | LR: {5:.06f} | Sigma: {6:.02f} | dataT: {7:.05f} | netT: {8:.05f} | ETA: {9:02d}:{10:02d} | ErrCam: {11:.06f} '.format(
                        opt.name, epoch, train_idx, len(train_data_loader), error.item(), lr, opt.sigma,
                                                                            iter_start_time - iter_data_time,
                                                                            iter_net_time - iter_start_time, int(eta // 60),
                        int(eta - 60 * (eta // 60)), errorCam))

                print(
                    'Color: {0} | Epoch: {1} | {2}/{3} | Err: {4:.06f} | LR: {5:.06f} | dataT: {6:.05f} | netT: {7:.05f} | ETA: {8:02d}:{9:02d}'.format(
                        opt.name, epoch, train_idx, len(train_data_loader),
                        errorC.item(),
                        lr,
                        iter_start_time - iter_data_time,
                        iter_net_time - iter_start_time, int(eta // 60),
                        int(eta - 60 * (eta // 60))))

            if train_idx % opt.freq_save == 0 and train_idx != 0:
                state_dict = {
                    'epoch': epoch,
                    'netG': netG.state_dict(),
                    'netCam': netCam.state_dict(),
                    'netC': netC.state_dict(),
                    'optimizerG': optimizerG.state_dict(),
                    'optimizerCam': optimizerCam.state_dict(),
                    'optimizerC': optimizerC.state_dict(),
                }
                torch.save(state_dict, '%s/%s/netGandCam_latest' % (opt.checkpoints_path, opt.name))


            if train_idx % opt.freq_save_ply == 0:
                save_path = '%s/%s/pred.ply' % (opt.results_path, opt.name)
                r = res[0].cpu()
                points = sample_tensor[0].transpose(0, 1).cpu()
                save_samples_truncted_prob(save_path, points.detach().numpy(), r.detach().numpy())

            iter_data_time = time.time()

        # update learning rate
        lr = adjust_learning_rate(optimizerG, epoch, lr, opt.schedule, opt.gamma)
        lrCam = adjust_learning_rate(optimizerCam, epoch, lrCam, opt.schedule, opt.gamma)

        #### test
        with torch.no_grad():
            set_eval()

            if not opt.no_num_eval:
                test_losses = {}
                print('calc error (test) ...')
                test_errors = calc_error(opt, netG, cuda, test_dataset, 100)
                print('eval test MSE: {0:06f} IOU: {1:06f} prec: {2:06f} recall: {3:06f}'.format(*test_errors))
                test_color_error = calc_error_color(opt, netG, netC, cuda, test_dataset, 100)
                print('eval test | color error:', test_color_error)
                test_losses['test_color'] = test_color_error

                MSE, IOU, prec, recall = test_errors
                test_losses['MSE(test)'] = MSE
                test_losses['IOU(test)'] = IOU
                test_losses['prec(test)'] = prec
                test_losses['recall(test)'] = recall

                print('calc error (train) ...')
                train_dataset.is_train = False
                train_errors = calc_error(opt, netG, cuda, train_dataset, 100)
                train_color_error = calc_error_color(opt, netG, netC, cuda, train_dataset, 100)
                train_dataset.is_train = True
                print('eval train MSE: {0:06f} IOU: {1:06f} prec: {2:06f} recall: {3:06f}'.format(*train_errors))
                print('eval train | color error:', train_color_error)
                MSE, IOU, prec, recall = train_errors
                test_losses['MSE(train)'] = MSE
                test_losses['IOU(train)'] = IOU
                test_losses['prec(train)'] = prec
                test_losses['recall(train)'] = recall
                test_losses['train_color'] = train_color_error

                writer_validation.add_scalar("LossG/validation", MSE, epoch)
                writer_validation.add_scalar("LossC/validation", train_color_error, epoch)

            if not opt.no_gen_mesh:
                print('generate mesh (test) ...')
                for gen_idx in tqdm(range(opt.num_gen_mesh_test)):
                    test_data = random.choice(test_dataset)
                    save_path = '%s/%s/test_eval_epoch%d_%s.obj' % (
                        opt.results_path, opt.name, epoch, test_data['name'])
                    #gen_mesh(opt, netG, cuda, test_data, save_path)
                    gen_mesh_color(opt, netG, netC, cuda, test_data, save_path)

                print('generate mesh (train) ...')
                train_dataset.is_train = False
                for gen_idx in tqdm(range(opt.num_gen_mesh_test)):
                    train_data = random.choice(train_dataset)
                    save_path = '%s/%s/train_eval_epoch%d_%s.obj' % (
                        opt.results_path, opt.name, epoch, train_data['name'])
                    #gen_mesh(opt, netG, cuda, train_data, save_path)
                    gen_mesh_color(opt, netG, netC, cuda, train_data, save_path)
                train_dataset.is_train = True

    writer.flush()
    writer.close()

if __name__ == '__main__':
    train(opt)