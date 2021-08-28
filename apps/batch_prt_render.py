import argparse
#from data.config import raw_dataset, render_dataset, archive_dataset, model_list, zip_path

from lib.renderer.camera import Camera
import numpy as np
from lib.renderer.mesh import load_obj_mesh, compute_tangent, compute_normal, load_obj_mesh_mtl
from lib.renderer.camera import Camera
from apps.prt_util import testPRT
from apps.render_data import render_prt_ortho

if __name__ == '__main__':
    for i in range(19):
        print('Generate prt for horse_z' + str(i))
        testPRT('./original_dataset/horse_z' + str(i))

        print('Render for horse_z' + str(i))
        shs = np.load('./env_sh.npy')

        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--input', type=str, default='./original_dataset/horse_z' + str(i))
        parser.add_argument('-o', '--out_dir', type=str, default='./training_horse')
        parser.add_argument('-m', '--ms_rate', type=int, default=1,
                            help='higher ms rate results in less aliased output. MESA renderer only supports ms_rate=1.')
        parser.add_argument('-e', '--egl', action='store_true',
                            help='egl rendering option. use this when rendering with headless server with NVIDIA GPU')
        parser.add_argument('-s', '--size', type=int, default=512, help='rendering image size')
        args = parser.parse_args()

        # NOTE: GL context has to be created before any other OpenGL function loads.
        from lib.renderer.gl.init_gl import initialize_GL_context

        initialize_GL_context(width=args.size, height=args.size, egl=args.egl)

        from lib.renderer.gl.prt_render import PRTRender

        rndr = PRTRender(width=args.size, height=args.size, ms_rate=args.ms_rate, egl=args.egl)
        rndr_uv = PRTRender(width=args.size, height=args.size, uv_mode=True, egl=args.egl)

        if args.input[-1] == '/':
            args.input = args.input[:-1]
        subject_name = args.input.split('/')[-1]
        render_prt_ortho(args.out_dir, args.input, subject_name, shs, rndr, rndr_uv, args.size, 1, 1, pitch=[0])