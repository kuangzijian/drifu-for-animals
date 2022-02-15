from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    TexturesVertex,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    AlphaCompositor, SoftSilhouetteShader, BlendParams
)
from .mesh_util import *
from .sample_util import *
from .geometry import *
import cv2
from PIL import Image
from tqdm import tqdm
from .sdf import create_point_cloud_grid

def reshape_multiview_tensors(image_tensor, calib_tensor, mask_tensor):
    # Careful here! Because we put single view and multiview together,
    # the returned tensor.shape is 5-dim: [B, num_views, C, W, H]
    # So we need to convert it back to 4-dim [B*num_views, C, W, H]
    # Don't worry classifier will handle multi-view cases
    if image_tensor != None:
        image_tensor = image_tensor.view(
            image_tensor.shape[0] * image_tensor.shape[1],
            image_tensor.shape[2],
            image_tensor.shape[3],
            image_tensor.shape[4]
        )

    if calib_tensor != None:
        calib_tensor = calib_tensor.view(
            calib_tensor.shape[0] * calib_tensor.shape[1],
            calib_tensor.shape[2],
            calib_tensor.shape[3]
        )

    if mask_tensor != None:
        mask_tensor = mask_tensor.view(
            mask_tensor.shape[0] * mask_tensor.shape[1],
            mask_tensor.shape[2],
            mask_tensor.shape[3],
            mask_tensor.shape[4]
        )
    return image_tensor, calib_tensor, mask_tensor


def reshape_sample_tensor(sample_tensor, num_views):
    if num_views == 1:
        return sample_tensor
    # Need to repeat sample_tensor along the batch dim num_views times
    sample_tensor = sample_tensor.unsqueeze(dim=1)
    sample_tensor = sample_tensor.repeat(1, num_views, 1, 1)
    sample_tensor = sample_tensor.view(
        sample_tensor.shape[0] * sample_tensor.shape[1],
        sample_tensor.shape[2],
        sample_tensor.shape[3]
    )
    return sample_tensor


def gen_mesh(opt, net, cuda, data, save_path, use_octree=True):
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)

    net.filter(image_tensor)

    b_min = data['b_min']
    b_max = data['b_max']
    try:
        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor.shape[0]):
            save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path)

        verts, faces, _, _ = reconstruction(
            net, cuda, calib_tensor, opt.resolution, b_min, b_max, use_octree=use_octree)
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()
        xyz_tensor = net.projection(verts_tensor, calib_tensor[:1])
        uv = xyz_tensor[:, :2, :]
        color = index(image_tensor[:1], uv).detach().cpu().numpy()[0].T
        color = color * 0.5 + 0.5
        save_obj_mesh_with_color(save_path, verts, faces, color)
    except Exception as e:
        print(e)
        print('Can not create marching cubes at this time.')

def gen_mesh_color(opt, netG, netC, cuda, data, save_path, use_octree=True, main_view_only=False):
    if main_view_only == True:
        image_tensor = data['img'][0].unsqueeze(0).to(device=cuda)
        calib_tensor = data['calib'][0].unsqueeze(0).to(device=cuda)
    else:
        image_tensor = data['img'].to(device=cuda)
        calib_tensor = data['calib'].to(device=cuda)
    if len(image_tensor.size()) == 5:
        image_tensor = image_tensor.squeeze(0)

    netG.filter(image_tensor)
    netC.filter(image_tensor)
    netC.attach(netG.get_im_feat())

    b_min = data['b_min']
    b_max = data['b_max']
    try:
        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor.shape[0]):
            save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path)

        verts, faces, _, _ = reconstruction(
            netG, cuda, calib_tensor, opt.resolution, b_min, b_max, use_octree=use_octree)

        # Now Getting colors
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()
        verts_tensor = reshape_sample_tensor(verts_tensor, opt.num_views)
        color = np.zeros(verts.shape)
        interval = 10000
        for i in range(len(color) // interval):
            left = i * interval
            right = i * interval + interval
            if i == len(color) // interval - 1:
                right = -1
            netC.query(verts_tensor[:, :, left:right], calib_tensor)
            rgb = netC.get_preds()[0].detach().cpu().numpy() * 0.5 + 0.5
            color[left:right] = rgb.T

        save_obj_mesh_with_color(save_path, verts, faces, color)
    except Exception as e:
        print(e)
        print('Can not create marching cubes at this time.')


def gen_point_cloud_color(opt, netG, netC, cuda, data, save_path, use_octree=True):
    image_tensor = data['img'].to(device=cuda)
    calib_tensor = data['calib'].to(device=cuda)
    b_min = data['b_min']
    b_max = data['b_max']
    with torch.no_grad():
        netG.filter(image_tensor)
        netC.filter(image_tensor)
        netC.attach(netG.get_im_feat())

    try:
        # Generate point cloud objects
        coords, mat = create_point_cloud_grid(opt.resolution, opt.resolution, opt.resolution,
                                         b_min, b_max, transform=None)
        points = np.expand_dims(coords, axis=0)
        points = np.repeat(points, netG.num_views, axis=0)
        samples = torch.from_numpy(points).to(device=cuda).float()
        netG.query(samples, calib_tensor)
        pred = netG.get_preds()[0][0]
        save_path = '../results/horse_2_test/stage2_pred.ply'
        points = samples[0].transpose(0, 1)
        netC.filter(image_tensor)
        netC.attach(netG.get_im_feat())
        new_points = get_positive_samples(save_path, points.detach().cpu().numpy(), pred.detach().cpu().numpy())
        new_samples = torch.from_numpy(new_points).to(device=cuda).float()
        netC.query(new_samples.T.unsqueeze(0), calib_tensor)
        pred_rgb = netC.get_preds()[0]
        rgb = pred_rgb.transpose(0, 1).cpu() * 0.5 + 0.5
        save_path_rgb = '../results/horse_2_test/stage2_pred_col.ply'
        save_samples_rgb(save_path_rgb, new_samples.detach().cpu().numpy(), rgb.detach().numpy())
        torch.cuda.empty_cache()

        point_cloud = Pointclouds(points=new_samples.unsqueeze(0), features=pred_rgb.T.unsqueeze(0))

        R, T = look_at_view_transform(dist=22.0, elev=0, azim=0, device=cuda)
        cameras = FoVOrthographicCameras(device=cuda, R=R, T=T, znear=0.01)

        # Define the settings for rasterization and shading. Here we set the output image to be of size
        # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
        # and blur_radius=0.0. Refer to raster_points.py for explanations of these parameters.
        raster_settings = PointsRasterizationSettings(
            image_size=512,
            radius=0.003,
            points_per_pixel=10
        )

        # Create a points renderer by compositing points using an alpha compositor (nearer points
        # are weighted more heavily). See [1] for an explanation.
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor()
        )

        pred_image_tensor = renderer(point_cloud)
        images_w_tex = np.clip(pred_image_tensor[0, ..., :3].cpu().numpy(), 0.0, 1.0)[:, :, ::-1] * 255
        cv2.imwrite('../results/horse_2_test/stage2_render_point_cloud.jpg', images_w_tex)

        # Generate mesh based object
        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor.shape[0]):
            save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path)
        verts, faces, _, _ = reconstruction(
            netG, cuda, calib_tensor, opt.resolution, b_min, b_max, use_octree=use_octree)

        # Now Getting colors
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()
        verts_tensor = reshape_sample_tensor(verts_tensor, opt.num_views)
        color = np.zeros(verts.shape)
        interval = 10000
        for i in range(len(color) // interval):
            left = i * interval
            right = i * interval + interval
            if i == len(color) // interval - 1:
                right = -1
            netC.query(verts_tensor[:, :, left:right], calib_tensor)
            rgb = netC.get_preds()[0].detach().cpu().numpy() * 0.5 + 0.5
            color[left:right] = rgb.T

        save_obj_mesh_with_color(save_path, verts, faces, color)
    except Exception as e:
        print(e)
        print('Can not create marching cubes at this time.')

def gen_mesh_color_tester(opt, netG, netC, cuda, data, calib_tensor, b_min, b_max, save_path, use_octree=True):
    image_tensor = data['img'].to(device=cuda)
    if len(image_tensor.size()) == 5:
        image_tensor = image_tensor.squeeze(0)

    try:
        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor.shape[0]):
            save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        Image.fromarray(np.uint8(save_img[:,:,::-1])).save(save_img_path)

        verts, faces, _, _ = reconstruction(
            netG, cuda, calib_tensor, opt.resolution, b_min, b_max, use_octree=use_octree)

        # Now Getting colors
        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=cuda).float()
        verts_tensor = reshape_sample_tensor(verts_tensor, opt.num_views)
        color = np.zeros(verts.shape)
        interval = 10000
        for i in range(len(color) // interval):
            left = i * interval
            right = i * interval + interval
            if i == len(color) // interval - 1:
                right = -1
            netC.query(verts_tensor[:, :, left:right], calib_tensor)
            rgb = netC.get_preds()[0].detach().cpu().numpy() * 0.5 + 0.5
            color[left:right] = rgb.T

        save_obj_mesh_with_color(save_path, verts, faces, color)
    except Exception as e:
        print(e)
        print('Can not create marching cubes at this time.')
    return verts, faces, color

def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def compute_acc(pred, gt, thresh=0.5):
    '''
    return:
        IOU, precision, and recall
    '''
    with torch.no_grad():
        vol_pred = pred > thresh
        vol_gt = gt > thresh

        union = vol_pred | vol_gt
        inter = vol_pred & vol_gt

        true_pos = inter.sum().float()

        union = union.sum().float()
        if union == 0:
            union = 1
        vol_pred = vol_pred.sum().float()
        if vol_pred == 0:
            vol_pred = 1
        vol_gt = vol_gt.sum().float()
        if vol_gt == 0:
            vol_gt = 1
        return true_pos / union, true_pos / vol_pred, true_pos / vol_gt

def compute_f1score(pred, gt, thresh=0.5):
    '''
    return:
        IOU, precision, and recall
    '''
    with torch.no_grad():
        vol_pred = pred > thresh
        vol_gt = gt > thresh

        union = vol_pred | vol_gt
        inter = vol_pred & vol_gt

        true_pos = inter.sum().float()

        union = union.sum().float()
        if union == 0:
            union = 1
        vol_pred = vol_pred.sum().float()
        if vol_pred == 0:
            vol_pred = 1
        vol_gt = vol_gt.sum().float()
        if vol_gt == 0:
            vol_gt = 1
        return true_pos / union

def calc_error(opt, net, cuda, dataset, num_tests, main_view_only=False):
    if num_tests > len(dataset):
        num_tests = len(dataset)
    with torch.no_grad():
        erorr_arr, IOU_arr, prec_arr, recall_arr = [], [], [], []
        for idx in tqdm(range(num_tests)):
            data = dataset[idx * len(dataset) // num_tests]
            # retrieve the data
            if main_view_only == True:
                image_tensor = data['img'][0].unsqueeze(0).to(device=cuda)
                calib_tensor = data['calib'][0].unsqueeze(0).to(device=cuda)
            else:
                image_tensor = data['img'].to(device=cuda)
                calib_tensor = data['calib'].to(device=cuda)
            sample_tensor = data['samples'].to(device=cuda).unsqueeze(0)
            if opt.num_views > 1:
                sample_tensor = reshape_sample_tensor(sample_tensor, opt.num_views)
            label_tensor = data['labels'].to(device=cuda).unsqueeze(0)

            res, error = net.forward(image_tensor, sample_tensor, calib_tensor, labels=label_tensor)

            IOU, prec, recall = compute_acc(res, label_tensor)

            # print(
            #     '{0}/{1} | Error: {2:06f} IOU: {3:06f} prec: {4:06f} recall: {5:06f}'
            #         .format(idx, num_tests, error.item(), IOU.item(), prec.item(), recall.item()))
            erorr_arr.append(error.item())
            IOU_arr.append(IOU.item())
            prec_arr.append(prec.item())
            recall_arr.append(recall.item())

    return np.average(erorr_arr), np.average(IOU_arr), np.average(prec_arr), np.average(recall_arr)

def calc_2d_error(opt, netC, netG, cuda, dataset, num_tests, save_path):
    if num_tests > len(dataset):
        num_tests = len(dataset)
    with torch.no_grad():
        IOU_arr, prec_arr, recall_arr = [], [], []
        for idx in tqdm(range(num_tests)):
            data = dataset[idx * len(dataset) // num_tests]
            # retrieve the data
            image_tensor = data['img'].to(device=cuda)
            mask_tensor = data['mask'].to(device=cuda)
            name = data['name']
            save_path = '../results/bird_3_test/' + name + '.obj'

            B_MIN = np.array([-1, -1, -1])
            B_MAX = np.array([1, 1, 1])
            projection_matrix = np.identity(4)
            projection_matrix[1, 1] = -1
            calib = torch.Tensor(projection_matrix).float().unsqueeze(0).to(device=cuda)

            netG.filter(image_tensor)
            netC.filter(image_tensor)
            netC.attach(netG.get_im_feat())
            verts, faces, color = gen_mesh_color_tester(opt, netG, netC, cuda, data, calib, B_MIN, B_MAX, save_path)
            verts_rgb_colors = torch.Tensor(color).to(cuda)
            verts = torch.Tensor(verts).to(cuda)
            faces = torch.from_numpy(faces.copy()).to(cuda)
            verts_list = []
            verts_list.append(verts)
            faces_list = []
            faces_list.append(faces)
            textures = TexturesVertex(verts_features=verts_rgb_colors.unsqueeze(0))
            mesh_w_tex = Meshes(verts_list, faces_list, textures)
            R, T = look_at_view_transform(dist=2.0, elev=0, azim=0, device=cuda)
            cameras = FoVOrthographicCameras(device=cuda, R=R, T=T)
            renderer, renderer_silhouette = set_renderer()
            rendered_mask = renderer_silhouette(mesh_w_tex, cameras=cameras)
            rendered_mask = rendered_mask[:,:,:,3].unsqueeze(0)
            rendered_mask = torch.where(rendered_mask>0, torch.ones(1, 1, 512, 512).to(cuda), torch.zeros(1, 1, 512, 512).to(cuda))
            #save_rendered_mask = (np.transpose(rendered_mask[0].detach().cpu().numpy(), (1, 2, 0)))[:, :, ::-1] * 255.0
            #Image.fromarray(np.uint8(save_rendered_mask[:, :, ::-1].repeat(3, axis=2))).save('../results/bird_3_test/stage2_test_render_mask.png')
            #save_mask = (np.transpose(mask_tensor[0].detach().cpu().numpy(), (1, 2, 0)))[:, :, ::-1] * 255.0
            #Image.fromarray(np.uint8(save_mask[:, :, ::-1].repeat(3, axis=2))).save('../results/bird_3_test/stage2_test_mask.png')

            IOU, prec, recall = compute_acc(rendered_mask, mask_tensor, 0)
            IOU_arr.append(IOU.item())
            prec_arr.append(prec.item())
            recall_arr.append(recall.item())

            # using SSIM for rgb image similarity check.
            rendered_img = renderer(mesh_w_tex, cameras=cameras)
            #save_rendered_img = rendered_img[:,:,:,0:3].squeeze(0).detach().cpu().numpy() * 255.0
            #Image.fromarray(np.uint8(save_rendered_img)).save('../results/bird_3_test/stage2_test_render_img.png')
            #save_img = (np.transpose(image_tensor[0].detach().cpu().numpy(), (1, 2, 0)))[:, :, ::-1] * 255.0
            #Image.fromarray(np.uint8(save_img)).save( '../results/bird_3_test/stage2_test_img.png')


    return np.average(IOU_arr), np.average(prec_arr), np.average(recall_arr)

def set_renderer():
    # Setup
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Initialize an OpenGL perspective camera.
    R, T = look_at_view_transform(dist=2.0, elev=0, azim=0, device=device)
    cameras = FoVOrthographicCameras(device=device, R=R, T=T)

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
            lights=lights,
            blend_params=BlendParams(background_color=(0.0, 0.0, 0.0))
        )
    )

    # Rasterization settings for silhouette rendering
    sigma = 1e-4
    raster_settings_silhouette = RasterizationSettings(
        image_size=512,
        blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
        faces_per_pixel=50,
    )

    # Silhouette renderer
    renderer_silhouette = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings_silhouette
        ),
        shader=SoftSilhouetteShader()
    )
    return renderer, renderer_silhouette

def calc_error_color(opt, netG, netC, cuda, dataset, num_tests, main_view_only=False):
    if num_tests > len(dataset):
        num_tests = len(dataset)
    with torch.no_grad():
        error_color_arr = []

        for idx in tqdm(range(num_tests)):
            data = dataset[idx * len(dataset) // num_tests]
            # retrieve the data
            if main_view_only == True:
                image_tensor = data['img'][0].unsqueeze(0).to(device=cuda)
                calib_tensor = data['calib'][0].unsqueeze(0).to(device=cuda)
            else:
                image_tensor = data['img'].to(device=cuda)
                calib_tensor = data['calib'].to(device=cuda)
            color_sample_tensor = data['color_samples'].to(device=cuda).unsqueeze(0)

            if opt.num_views > 1:
                color_sample_tensor = reshape_sample_tensor(color_sample_tensor, opt.num_views)

            rgb_tensor = data['rgbs'].to(device=cuda).unsqueeze(0)

            netG.filter(image_tensor)
            _, errorC = netC.forward(image_tensor, netG.get_im_feat(), color_sample_tensor, calib_tensor, labels=rgb_tensor)

            # print('{0}/{1} | Error inout: {2:06f} | Error color: {3:06f}'
            #       .format(idx, num_tests, errorG.item(), errorC.item()))
            error_color_arr.append(errorC.item())

    return np.average(error_color_arr)

