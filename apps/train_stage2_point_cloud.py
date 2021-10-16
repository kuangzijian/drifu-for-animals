import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import json
from torch.utils.data import DataLoader

from lib.options import BaseOptions
from lib.train_util import *
from lib.data import *
from lib.model import *
from lib.mesh_util import *
from lib.sdf import *
from apps.render_data import *
from apps.prt_util import *

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointsRasterizer,
    PointsRenderer,
    PointsRasterizationSettings,
    AlphaCompositor
)

# get options
opt = BaseOptions().parse()

def train_stage2(opt):
    # set cuda
    cuda = torch.device('cuda:%d' % opt.gpu_id)

    train_dataset = TrainDataset_Stage2(opt, phase='train')
    test_dataset = TrainDataset_Stage2(opt, phase='test')

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

    # load 3 pretrained networks
    resume_path = '%s/%s/netGandCam_latest' % (opt.checkpoints_path, opt.name)
    checkpoint = torch.load(resume_path)
    print('Resuming from ', resume_path)
    netG.load_state_dict(checkpoint['netG'])
    netCam.load_state_dict(checkpoint['netCam'])
    netC.load_state_dict(checkpoint['netC'])
    resume_epoch = checkpoint['epoch']

    import os
    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

    opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
    with open(opt_log, 'w') as outfile:
        outfile.write(json.dumps(vars(opt), indent=2))

    # freeze the encoder/decoders and retrain the MLPs

    # training
    start_epoch = 0 if not opt.continue_train else resume_epoch
    for epoch in range(start_epoch, opt.stage2_num_epoch):
        epoch_start_time = time.time()

        set_train()
        iter_data_time = time.time()
        for train_idx, train_data in enumerate(train_data_loader):
            iter_start_time = time.time()

            # retrieve the data
            image_tensor = train_data['img'].to(device=cuda)
            mask_tensor = train_data['mask'].to(device=cuda)
            B_MIN = np.array([-1, -1, -1])
            B_MAX = np.array([1, 1, 1])
            projection_matrix = np.identity(4)
            projection_matrix[1, 1] = -1
            calib = torch.Tensor(projection_matrix).float().unsqueeze(0).to(device=cuda)

            image_tensor = image_tensor.view(
                image_tensor.shape[0] * image_tensor.shape[1],
                image_tensor.shape[2],
                image_tensor.shape[3],
                image_tensor.shape[4]
            )

            # generate 3D mesh info from 2D image
            coords, mat = create_point_cloud_grid_tensor(opt.resolution, opt.resolution, opt.resolution,
                                             B_MIN, B_MAX, transform=None)
            points = np.expand_dims(coords, axis=0)
            points = np.repeat(points, netG.num_views, axis=0)
            samples = torch.from_numpy(points).to(device=cuda).float()
            netG.filter(image_tensor)
            netG.query(samples, calib)
            pred = netG.get_preds()[0][0]
            save_path = '../results/horse_2_test/stage2_pred.ply'
            points = samples[0].transpose(0, 1)
            netC.filter(image_tensor)
            netC.attach(netG.get_im_feat())
            new_points = get_positive_samples(save_path, points.detach().cpu().numpy(), pred.detach().cpu().numpy())
            new_samples = torch.from_numpy(new_points).to(device=cuda).float()
            netC.query(new_samples.T.unsqueeze(0), calib)
            pred_rgb = netC.get_preds()[0]
            rgb = pred_rgb.transpose(0, 1).cpu() * 0.5 + 0.5
            save_path_rgb = '../results/horse_2_test/stage2_pred_col.ply'
            save_samples_rgb(save_path_rgb, new_samples.detach().cpu().numpy(), rgb.detach().numpy())

            # generate obj file for testing
            save_path = '../results/horse_2_test/stage2.obj'
            gen_mesh_color_tester(opt, netG, netC, cuda, train_data, calib, B_MIN, B_MAX, save_path)

            # render 2D image from 3D info
            renderer = set_renderer(cuda)
            point_cloud = Pointclouds(points=new_samples.unsqueeze(0), features=pred_rgb.T.unsqueeze(0))
            pred_image_tensor = renderer(point_cloud)
            images_w_tex = np.clip(pred_image_tensor[0, ..., :3].detach().cpu().numpy(), 0.0, 1.0)[:, :, ::-1] * 255
            cv2.imwrite('../results/horse_2_test/stage2_render_point_cloud.jpg', images_w_tex)

            # get 2D supervision loss
            loss_image = torch.mean(torch.abs(pred_image_tensor.permute(0,3,1,2) - image_tensor))

            optimizerG.zero_grad()
            optimizerC.zero_grad()
            loss_image.backward()
            optimizerG.step()
            optimizerC.step()

            iter_net_time = time.time()
            eta = ((iter_net_time - epoch_start_time) / (train_idx + 1)) * len(train_data_loader) - (
                    iter_net_time - epoch_start_time)

            if train_idx % opt.freq_plot == 0:
                print(
                    'Stage2: {0} | Epoch: {1} | {2}/{3} | Err: {4:.06f} | LR: {5:.06f} | Sigma: {6:.02f} | dataT: {7:.05f} | netT: {8:.05f} | ETA: {9:02d}:{10:02d}'.format(
                        opt.name, epoch, train_idx, len(train_data_loader), loss_image.item(), lr, opt.sigma,
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

            iter_data_time = time.time()

        # update learning rate
        lr = adjust_learning_rate(optimizerG, epoch, lr, opt.schedule, opt.gamma)

def set_renderer(cuda):
    # Setup
    R, T = look_at_view_transform(dist=22.0, elev=0, azim=0, device=cuda)
    cameras = FoVOrthographicCameras(device=cuda, R=R, T=T, znear=0.01)
    raster_settings = PointsRasterizationSettings(
        image_size=512,
        radius=0.003,
        points_per_pixel=10
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    )

    return renderer

if __name__ == '__main__':
    train_stage2(opt)