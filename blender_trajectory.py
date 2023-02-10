import sys
import os

sys.path.append(os.path.dirname(__file__))  # So blender's python can find this folder
import bpy
from pathlib import Path
from argparse import ArgumentParser
import blender.blender_utils as butils
import debugpy
from collections import defaultdict
import json
import numpy as np


def start_debugger():
    debugpy.listen(5678)
    print("Waiting for debugger to attach... ", end='', flush=True)
    debugpy.wait_for_client()
    print("done!")


bladder_name = 'labels-81_bladder_smooth_000'
bladder_material_names = ['bladder_wall', 'bland_bladder_wall']
camera_name = 'CamFrom3x3PObj'

if __name__ == '__main__':
    arguments, headless = butils.extract_system_arguments()
    parser = ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Will start the remote debugging on port 5678.')
    parser.add_argument('--trajectory_file', help='path to blender file with trajectory.')
    parser.add_argument('--output_directory', help='directory to export trajectory to.')
    parser.add_argument('--gpu', type=int, default=-1, help='specify gpu to use. defaults to all available')
    parser.add_argument('--gpu_type', type=str, default='METAL', help='one of [CUDA, OPTIX, METAL, OPENCL]')
    args, unknown_args = parser.parse_known_args(arguments)

    butils.set_gpu_rendering_preferences(args.gpu, device_type=args.gpu_type)

    output_dir = Path(args.output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_file = Path(args.trajectory_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Seems like {input_file} doesn't exist. :( ")
    bpy.ops.wm.open_mainfile(filepath=str(input_file))
    context = bpy.context
    scene = context.scene
    color_node = scene.node_tree.nodes['color_out']
    depth_node = scene.node_tree.nodes['depth_out']
    normals_node = scene.node_tree.nodes['normals_out']
    output_nodes = [color_node, depth_node, normals_node]
    path = bpy.data.objects.get('camera_path')
    number_of_frames = path.data.path_duration
    camera = bpy.data.objects.get(camera_name)
    bladder = bpy.data.objects.get(bladder_name)
    [setattr(n.file_slots[0], 'path', f'frame_######') for n in output_nodes if n is not None]

    depth_node.base_path = os.path.join(str(output_dir), 'depth')
    normals_node.base_path = os.path.join(str(output_dir), 'normals')

    camera_trajectory = defaultdict(dict)
    number_of_frames = 2
    for frame_number in range(1, number_of_frames + 1):
        scene.frame_set(frame_number)
        print(frame_number)
        camera_trajectory[frame_number]['extrinsic_matrix'] = np.asarray(camera.matrix_world).tolist()
        camera_trajectory[frame_number]['depth'] = str(Path('.', 'depth', f'frame_{frame_number:06d}.exr'))
        camera_trajectory[frame_number]['normals'] = str(Path('.', 'normals', f'frame_{frame_number:06d}.exr'))
        camera_trajectory[frame_number]['color'] = {}
        scene.render.engine = 'CYCLES'
        for bladder_mat in bladder_material_names:
            camera_trajectory[frame_number]['color'][bladder_mat] = str(
                Path('.', 'colors', bladder_mat, f'frame_{frame_number:06d}.exr'))

            depth_node.mute = True
            normals_node.mute = True
            color_node.mute = False
            color_node.base_path = os.path.join(str(output_dir), 'color', bladder_mat)
            bladder.material_slots[0].material = bpy.data.materials[bladder_mat]
            bpy.ops.render.render(write_still=True, scene=scene.name)

        depth_node.mute = False
        normals_node.mute = False
        color_node.mute = True
        # switch to basic material and renderer for rendering normals and depth
        scene.render.engine = 'BLENDER_EEVEE'
        bladder.material_slots[0].material = bpy.data.materials['Material']
        bpy.ops.render.render(write_still=True, scene=scene.name)

    json.dump(camera_trajectory, open(str(Path(output_dir, 'trajectory.json')), 'w'))
