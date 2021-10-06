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
from apps.render_data import *
from apps.prt_util import *

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    TexturesVertex
)

# get options
opt = BaseOptions().parse()

def train_stage2(opt):
    # set cuda
    cuda = torch.device('cuda:%d' % opt.gpu_id)

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
        netCam.eval()
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
            sample_tensor = train_data['samples'].to(device=cuda)
            calib_tensor = train_data['calib'].to(device=cuda)
            camera_tensor = train_data['camera'].to(device=cuda)
            color_sample_tensor = train_data['color_samples'].to(device=cuda)
            B_MIN = np.array([-1, -1, -1])
            B_MAX = np.array([1, 1, 1])
            projection_matrix = np.identity(4)
            projection_matrix[1, 1] = -1
            calib = torch.Tensor(projection_matrix).float().unsqueeze(0).to(device=cuda)

            image_tensor, calib_tensor = reshape_multiview_tensors(image_tensor, calib_tensor)

            label_tensor = train_data['labels'].to(device=cuda)
            rgb_tensor = train_data['rgbs'].to(device=cuda)

            res, error = netG.forward(image_tensor, sample_tensor, calib, labels=label_tensor)
            resCam, errorCam = netCam.forward(image_tensor, camera_tensor)

            with torch.no_grad():
                netG.filter(image_tensor)
            resC, errorC = netC.forward(image_tensor, netG.get_im_feat(), color_sample_tensor, calib, labels=rgb_tensor)

            # generate 3D mesh info from 2D image
            verts, faces, normals, values = reconstruction(
                netG, cuda, calib, opt.resolution, B_MIN, B_MAX, use_octree=True)

            # modify 3D mesh data format
            verts = verts.astype('float32')
            for idx, f in enumerate(faces):
                faces[idx] = [f[0], f[2], f[1]]

            # generate 3D colors from 2D image
            verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()
            verts_tensor = reshape_sample_tensor(verts_tensor, opt.num_views)
            colors = np.zeros(verts.shape)
            interval = 10000
            for i in range(len(colors) // interval):
                left = i * interval
                right = i * interval + interval
                if i == len(colors) // interval - 1:
                    right = -1
                netC.query(verts_tensor[:, :, left:right], calib)
                rgb = netC.get_preds()[0].detach().cpu().numpy() * 0.5 + 0.5
                colors[left:right] = rgb.T

            # generate obj file for testing
            train_data_temp = train_data
            save_path = '../results/horse_2_test/stage2.obj'
            gen_mesh_color_tester(opt, netG, netC, cuda, train_data_temp, calib, B_MIN, B_MAX, save_path)
            #save_obj_mesh_with_color_tester(save_path, verts, faces, colors)

            # render 2D image from 3D info
            render = set_renderer()
            image = render_func(verts, faces, colors, render, cuda)

            # get 2D supervision loss based on weighted image and mask losses
            image_weight = 0.1
            mask_weight = 1.
            loss_image = torch.mean(torch.abs(image - image_tensor))
            loss_mask, _, _ = compute_acc(image, image_tensor)
            loss = image_weight * loss_image + mask_weight * loss_mask

            optimizerG.zero_grad()
            optimizerCam.zero_grad()
            optimizerC.zero_grad()
            loss.backward()
            optimizerG.step()
            optimizerCam.step()
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

            if not opt.no_gen_mesh:
                print('generate mesh (test) ...')
                for gen_idx in tqdm(range(opt.num_gen_mesh_test)):
                    test_data = random.choice(test_dataset)
                    save_path = '%s/%s/test_eval_epoch%d_%s.obj' % (
                        opt.results_path, opt.name, epoch, test_data['name'])
                    gen_mesh_color(opt, netG, netC, cuda, test_data, save_path)

                print('generate mesh (train) ...')
                train_dataset.is_train = False
                for gen_idx in tqdm(range(opt.num_gen_mesh_test)):
                    train_data = random.choice(train_dataset)
                    save_path = '%s/%s/train_eval_epoch%d_%s.obj' % (
                        opt.results_path, opt.name, epoch, train_data['name'])
                    gen_mesh_color(opt, netG, netC, cuda, train_data, save_path)
                train_dataset.is_train = True

def set_renderer():
    # Setup
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Initialize an OpenGL perspective camera.
    R, T = look_at_view_transform(dist=2.0, elev=0, azim=0, device=device)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size = None,
        max_faces_per_bin = None
    )

    lights = PointLights(device=device, location=((2.0, 2.0, 2.0),))

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )
    return renderer

def render_func(verts, faces, colors, renderer, device):
    # Setup
    torch.cuda.set_device(device)
    colors = torch.from_numpy(colors).type(torch.float32).to(device)
    textures = TexturesVertex(verts_features=colors.unsqueeze(0))

    # Set mesh
    verts_tensor = torch.from_numpy(verts).type(torch.float32).to(device)
    faces_tensor = torch.from_numpy(faces.copy()).type(torch.int64).to(device)
    verts_list = []
    faces_list = []
    verts_list.append(verts_tensor.to(device))
    faces_list.append(faces_tensor.to(device))
    mesh_w_tex = Meshes(verts_list, faces_list, textures)
    # create image file
    R, T = look_at_view_transform(dist=2.0, elev=0, azim=0, device=device)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    images_w_tex = renderer(mesh_w_tex, cameras=cameras)
    images_w_tex = np.clip(images_w_tex[0, ..., :3].cpu().numpy(), 0.0, 1.0)[:, :, ::-1] * 255
    cv2.imwrite('../results/horse_2_test/stage2.jpg', images_w_tex)
    return images_w_tex

if __name__ == '__main__':
    train_stage2(opt)