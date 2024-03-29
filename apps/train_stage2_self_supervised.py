import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import json
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
    # set cuda (single gpu)
    cuda = torch.device('cuda:%d' % opt.gpu_id)

    # set cuda (multiple gpus)
    #cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()

    train_dataset = TrainDataset_Stage2(opt, phase='train')
    test_dataset = TrainDataset_Stage2(opt, phase='test')

    projection_mode = 'orthogonal'

    # create data loader
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=opt.batch_size, shuffle=not opt.serial_batches,
                                   pin_memory=opt.pin_memory)

    print('train data size: ', len(train_data_loader))

    # NOTE: batch size should be 1 and use all the points for evaluation
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=opt.batch_size, shuffle=False,
                                  pin_memory=opt.pin_memory)
    print('test data size: ', len(test_data_loader))

    # create shape net and color net
    netG = HGPIFuNet(opt, projection_mode).to(device=cuda)
    netC = ResBlkPIFuNet(opt).to(device=cuda)

    optimizerG = torch.optim.RMSprop(netG.parameters(), lr=opt.learning_rate, momentum=0, weight_decay=0)
    optimizerC = torch.optim.Adam(netC.parameters(), lr=opt.learning_rate)

    lr = opt.learning_rate
    print('Using Network: ', netG.name)
    print('Using Network: ', netC.name)

    def set_train():
        netG.train()
        netC.train()

    def set_eval():
        netG.eval()
        netC.eval()

    # load 3 pretrained networks
    resume_path = '%s/%s/netGandCam_latest' % (opt.checkpoints_path, opt.name)
    checkpoint = torch.load(resume_path)
    print('Resuming from ', resume_path)
    netG.load_state_dict(checkpoint['netG'])
    netC.load_state_dict(checkpoint['netC'])
    # freeze the encoder and retrain the MLPs
    for param in netG.image_filter.parameters():
        param.requires_grad = False
    for param in netC.image_filter.parameters():
        param.requires_grad = False

    resume_epoch = checkpoint['epoch']

    import os
    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(opt.results_path, exist_ok=True)
    os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name), exist_ok=True)
    os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

    opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
    with open(opt_log, 'w') as outfile:
        outfile.write(json.dumps(vars(opt), indent=2))

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
            name = train_data['name'][0]
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

            mask_tensor = mask_tensor.view(
                mask_tensor.shape[0] * mask_tensor.shape[1],
                mask_tensor.shape[2],
                mask_tensor.shape[3],
                mask_tensor.shape[4]
            )

            # save ground truth image and mask for intermediate result evaluation
            save_path = '../results/bird_3_test/train_' + name + '.obj'
            save_img_path = save_path[:-4] + '_img.png'
            save_img = (np.transpose(image_tensor[0].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :,
                       ::-1] * 255.0
            #Image.fromarray(np.uint8(save_img[:, :, ::-1])).save(save_img_path)

            save_mask_path = save_path[:-4] + '_mask.png'
            save_mask = (np.transpose(mask_tensor[0].detach().cpu().numpy(), (1, 2, 0)))[:, :,
                        ::-1] * 255.0
            #Image.fromarray(np.uint8(save_mask[:, :, ::-1].repeat(3, axis=2))).save(save_mask_path)


            # generate 3D mesh info from 2D image
            coords, _ = create_point_cloud_grid_tensor(opt.resolution, opt.resolution, opt.resolution,
                                             B_MIN, B_MAX, transform=None, sample=opt.num_sample_stage2)
            points = np.expand_dims(coords, axis=0)
            points = np.repeat(points, netG.num_views, axis=0)
            samples = torch.from_numpy(points).to(device=cuda).float()
            netG.filter(image_tensor)
            netG.query(samples, calib)
            pred = netG.get_preds()[0][0]
            points = samples[0].transpose(0, 1)

            # get positive samples only
            positive_samples = points[pred > 0.5]
            if positive_samples.shape[0] == 0:
                positive_samples = points[0].unsqueeze(0)

            # render 2D silhouettes from predicted 3D info
            renderer = set_renderer(cuda)
            point_cloud = Pointclouds(points=positive_samples.unsqueeze(0), features=torch.ones(positive_samples.shape).unsqueeze(0).to(device=cuda))
            pred_mask_tensor = renderer(point_cloud)
            #mask = np.clip(pred_mask_tensor[0, ..., :3].detach().cpu().numpy(), 0.0, 1.0)[:, :, ::-1] * 255
            #cv2.imwrite('../results/bird_3_test/stage2_render_mask.jpg', mask)

            # render 2D images from predicted 3D info
            netC.filter(image_tensor)
            netC.attach(netG.get_im_feat())

            # generate obj file for intermediate result evaluation
            #verts, faces, color = gen_mesh_color_tester(opt, netG, netC, cuda, train_data, calib, B_MIN, B_MAX, save_path)

            netC.query(positive_samples.T.unsqueeze(0), calib)
            pred_rgb = netC.get_preds()[0]
            rgb = pred_rgb.transpose(0, 1).cpu() * 0.5 + 0.5
            #save_path_rgb = '../results/bird_3_test/stage2_pred_col.ply'
            #save_samples_rgb(save_path_rgb, positive_samples.detach().cpu().numpy(), rgb.detach().numpy())

            point_cloud_colored = Pointclouds(points=positive_samples.unsqueeze(0), features=pred_rgb.T.unsqueeze(0))
            pred_image_tensor = renderer(point_cloud_colored)
            #images_w_tex = np.clip(pred_image_tensor[0, ..., :3].detach().cpu().numpy(), 0.0, 1.0)[:, :, ::-1] * 255
            #cv2.imwrite('../results/bird_3_test/stage2_render_img.jpg', images_w_tex)

            # get 2D supervision loss
            loss_image = torch.mean(torch.abs(pred_image_tensor.permute(0, 3, 1, 2) - image_tensor))
            loss_mask = torch.mean(torch.abs(pred_mask_tensor.permute(3, 0, 1, 2)[0].unsqueeze(0) - mask_tensor))
            loss = 0.3 * loss_image + 0.7 * loss_mask

            writer.add_scalar("Loss_img", loss_image, epoch)
            writer.add_scalar("Loss_mask", loss_mask, epoch)
            writer.add_scalar("loss", loss, epoch)

            optimizerG.zero_grad()
            optimizerC.zero_grad()
            loss.backward()
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
                    'netC': netC.state_dict(),
                    'optimizerG': optimizerG.state_dict(),
                    'optimizerC': optimizerC.state_dict(),
                }
                torch.save(state_dict, '%s/%s/netGandC_stage2_latest' % (opt.checkpoints_path, opt.name))

            iter_data_time = time.time()

        # update learning rate
        lr = adjust_learning_rate(optimizerG, epoch, lr, opt.schedule, opt.gamma)

        #### test
        with torch.no_grad():
            set_eval()

            if not opt.no_num_eval:
                test_losses = {}
                print('calc error (test) ...')
                save_path = '%s/%s/test_eval_epoch%d_%s.obj' % (opt.results_path, opt.name, epoch, 'test')
                test_errors = calc_2d_error(opt, netC, netG, cuda, test_dataset, 100, save_path)
                IOU, prec, recall, ssim = test_errors
                test_losses['IOU(test)'] = IOU
                test_losses['prec(test)'] = prec
                test_losses['recall(test)'] = recall
                test_losses['ssim'] = ssim

                print('eval test IOU: {0:06f} prec: {1:06f} recall: {2:06f} ssim: {3:06f}'.format(*test_errors))

    writer.flush()
    writer.close()

def set_renderer(cuda):
    # Setup
    R, T = look_at_view_transform(dist=22.0, elev=0, azim=0, device=cuda)
    cameras = FoVOrthographicCameras(device=cuda, R=R, T=T, znear=0.01)
    raster_settings = PointsRasterizationSettings(
        image_size=512,
        radius=0.015,
        points_per_pixel=100
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    )

    return renderer

if __name__ == '__main__':
    train_stage2(opt)