#from data.config import raw_dataset, render_dataset, archive_dataset, model_list, zip_path

from lib.renderer.camera import Camera
import numpy as np
from lib.renderer.mesh import load_obj_mesh, compute_tangent, compute_normal, load_obj_mesh_mtl
from lib.renderer.camera import Camera
import os
import cv2
import time
import math
import random
import pyexr
import argparse
from tqdm import tqdm
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,FoVOrthographicCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    TexturesVertex
)

def render_prt_ortho(out_path, folder_name, subject_name, rndr, rndr_uv, im_size, angl_step=4, n_light=1, pitch=[0]):
    cam = Camera(width=im_size, height=im_size)
    cam.ortho_ratio = 0.4 * (512 / im_size)
    cam.near = -100
    cam.far = 100
    cam.sanity_check()

    # set path for obj, prt
    mesh_file = os.path.join(folder_name, subject_name + '.obj')
    if not os.path.exists(mesh_file):
        print('ERROR: obj file does not exist!!', mesh_file)
        return 
    prt_file = os.path.join(folder_name, 'bounce', 'bounce0.txt')
    if not os.path.exists(prt_file):
        print('ERROR: prt file does not exist!!!', prt_file)
        return
    face_prt_file = os.path.join(folder_name, 'bounce', 'face.npy')
    if not os.path.exists(face_prt_file):
        print('ERROR: face prt file does not exist!!!', prt_file)
        return
    text_file = os.path.join(folder_name, 'tex', subject_name + '.png')
    if not os.path.exists(text_file):
        print('ERROR: dif file does not exist!!', text_file)
        return             

    texture_image = cv2.imread(text_file)
    texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)

    vertices, faces, normals, faces_normals, textures, face_textures = load_obj_mesh(mesh_file, with_normal=True, with_texture=True)
    vmin = vertices.min(0)
    vmax = vertices.max(0)
    up_axis = 1 if (vmax-vmin).argmax() == 1 else 2
    
    vmed = np.median(vertices, 0)
    vmed[up_axis] = 0.5*(vmax[up_axis]+vmin[up_axis])
    y_scale = 180/(vmax[up_axis] - vmin[up_axis])

    rndr.set_norm_mat(y_scale, vmed)
    rndr_uv.set_norm_mat(y_scale, vmed)

    tan, bitan = compute_tangent(vertices, faces, normals, textures, face_textures)
    prt = np.loadtxt(prt_file)
    face_prt = np.load(face_prt_file)
    rndr.set_mesh(vertices, faces, normals, faces_normals, textures, face_textures, prt, face_prt, tan, bitan)
    rndr.set_albedo(texture_image)

    rndr_uv.set_mesh(vertices, faces, normals, faces_normals, textures, face_textures, prt, face_prt, tan, bitan)   
    rndr_uv.set_albedo(texture_image)

    os.makedirs(os.path.join(out_path, 'GEO', 'OBJ', subject_name),exist_ok=True)
    os.makedirs(os.path.join(out_path, 'PARAM', subject_name),exist_ok=True)
    os.makedirs(os.path.join(out_path, 'RENDER', subject_name),exist_ok=True)
    os.makedirs(os.path.join(out_path, 'MASK', subject_name),exist_ok=True)
    os.makedirs(os.path.join(out_path, 'UV_RENDER', subject_name),exist_ok=True)
    os.makedirs(os.path.join(out_path, 'UV_MASK', subject_name),exist_ok=True)
    os.makedirs(os.path.join(out_path, 'UV_POS', subject_name),exist_ok=True)
    os.makedirs(os.path.join(out_path, 'UV_NORMAL', subject_name),exist_ok=True)

    if not os.path.exists(os.path.join(out_path, 'val.txt')):
        f = open(os.path.join(out_path, 'val.txt'), 'w')
        f.close()

    # copy obj file
    cmd = 'cp %s %s' % (mesh_file, os.path.join(out_path, 'GEO', 'OBJ', subject_name))
    print(cmd)
    os.system(cmd)

    for p in pitch:
        for y in tqdm(range(0, 360, angl_step)):
            R = np.matmul(make_rotate(math.radians(p), 0, 0), make_rotate(0, math.radians(y), 0))
            if up_axis == 2:
                R = np.matmul(R, make_rotate(math.radians(0),0,0))

            rndr.rot_matrix = R
            rndr_uv.rot_matrix = R
            rndr.set_camera(cam)
            rndr_uv.set_camera(cam)

            for j in range(n_light):
                sh_id = random.randint(0,shs.shape[0]-1)
                sh = shs[sh_id]
                sh_angle = 0.2*np.pi*(random.random()-0.5)
                sh = rotateSH(sh, make_rotate(0, sh_angle, 0).T)

                dic = {'sh': sh, 'ortho_ratio': cam.ortho_ratio, 'scale': y_scale, 'center': vmed, 'R': R}
                
                rndr.set_sh(sh)        
                rndr.analytic = False
                rndr.use_inverse_depth = False
                rndr.display()

                out_all_f = rndr.get_color(0)
                out_mask = out_all_f[:,:,3]
                out_all_f = cv2.cvtColor(out_all_f, cv2.COLOR_RGBA2BGR)

                np.save(os.path.join(out_path, 'PARAM', subject_name, '%d_%d_%02d.npy'%(y,p,j)),dic)
                cv2.imwrite(os.path.join(out_path, 'RENDER', subject_name, '%d_%d_%02d.jpg'%(y,p,j)),255.0*out_all_f)
                cv2.imwrite(os.path.join(out_path, 'MASK', subject_name, '%d_%d_%02d.png'%(y,p,j)),255.0*out_mask)

                rndr_uv.set_sh(sh)
                rndr_uv.analytic = False
                rndr_uv.use_inverse_depth = False
                rndr_uv.display()

                uv_color = rndr_uv.get_color(0)
                uv_color = cv2.cvtColor(uv_color, cv2.COLOR_RGBA2BGR)
                cv2.imwrite(os.path.join(out_path, 'UV_RENDER', subject_name, '%d_%d_%02d.jpg'%(y,p,j)),255.0*uv_color)

                if y == 0 and j == 0 and p == pitch[0]:
                    uv_pos = rndr_uv.get_color(1)
                    uv_mask = uv_pos[:,:,3]
                    cv2.imwrite(os.path.join(out_path, 'UV_MASK', subject_name, '00.png'),255.0*uv_mask)

                    data = {'default': uv_pos[:,:,:3]} # default is a reserved name
                    pyexr.write(os.path.join(out_path, 'UV_POS', subject_name, '00.exr'), data) 

                    uv_nml = rndr_uv.get_color(2)
                    uv_nml = cv2.cvtColor(uv_nml, cv2.COLOR_RGBA2BGR)
                    cv2.imwrite(os.path.join(out_path, 'UV_NORMAL', subject_name, '00.png'),255.0*uv_nml)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='../original_dataset/horse_a0')
    parser.add_argument('-o', '--out_dir', type=str, default='../training_dataset_animal')
    parser.add_argument('-m', '--ms_rate', type=int, default=1, help='higher ms rate results in less aliased output. MESA renderer only supports ms_rate=1.')
    parser.add_argument('-e', '--egl',  default=True, action='store_true', help='egl rendering option. use this when rendering with headless server with NVIDIA GPU')
    parser.add_argument('-s', '--size',  type=int, default=512, help='rendering image size')
    args = parser.parse_args()


    from lib.renderer.gl.prt_render import PRTRender
    rndr = PRTRender(width=args.size, height=args.size, ms_rate=args.ms_rate, egl=args.egl)
    rndr_uv = PRTRender(width=args.size, height=args.size, uv_mode=True, egl=args.egl)

    render = set_renderer()

    if args.input[-1] == '/':
        args.input = args.input[:-1]
    subject_name = args.input.split('/')[-1]
    render_prt_ortho(args.out_dir, args.input, subject_name, render, rndr_uv, args.size, 4, 1, pitch=[0])