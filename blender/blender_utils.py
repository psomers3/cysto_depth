import numpy as np
import os
import sys
import bpy
from config import BlenderConfig
from config.blender_config import ShrinkwrapConfig
from typing import *
import bmesh
from mathutils import Matrix, Vector, Euler, Quaternion


def random_unit_vectors(num_points: int, ndim: int = 3) -> np.ndarray:
    """
    Randomly generate unit vectors of dimension ndim

    :param num_points: number of vectors to generate
    :param ndim: dimension of vectors
    :returns: array of random unit vectors each of length ndim
    """
    vec = np.random.randn(ndim, num_points)
    vec /= np.linalg.norm(vec, axis=0)
    return vec.T


def init_blender(configuration: BlenderConfig) -> bpy.types.Scene:
    """ Initialize blender scene by deleting demo objects and applying settings from configuration

    :param configuration: A set of blender configurations to apply to the current scene
    :returns: the current active scene
    """
    context = bpy.context
    scene = context.scene
    for c in scene.collection.children:
        bpy.data.collections.remove(c, do_unlink=True)
    set_blender_data(scene, configuration)
    return scene


def set_blender_data(item: Any, config: Union[dict, BlenderConfig, Any]) -> None:
    """
    This function recursively sets all settings in a blender object assuming the dictionary entries are
    structured and named correctly.

    :param item: This should be a blender Object or Scene, etc.
    :param config: a dictionary-like object with blender settings
    """
    if hasattr(config, '__getitem__'):
        for key in config:
            if hasattr(config[key], '__getitem__') and not isinstance(config[key], str):
                set_blender_data(getattr(item, key), config[key])
            else:
                setattr(item, key, config[key])


def import_stl(stl_file: str,
               center: bool = False,
               smooth_shading: bool = True,
               collection: bpy.types.Collection = None) -> bpy.types.Object:
    """
    import an STL into blender

    :param stl_file: path to the STL file
    :param center: whether to move the stl's center of volume to the origin
    :param smooth_shading: sets the model to be displayed as smooth
    :param collection: a collection to put the object in
    :return: the blender Object for the STL mesh object
    """
    bpy.ops.import_mesh.stl(filepath=stl_file)
    obj = bpy.data.objects[os.path.splitext(os.path.basename(stl_file))[0]]
    if collection is not None:
        bpy.context.collection.objects.unlink(obj)
        collection.objects.link(obj)

    if center:
        obj.select_set(True)
        bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
        obj.location = [0, 0, 0]
    if smooth_shading:
        obj.select_set(True)
        bpy.ops.object.shade_smooth()
    obj.select_set(False)
    return obj


def scale_mesh_volume(obj: bpy.types.Object, volume: float) -> None:
    """
    scale a blender object by volume to a given volume

    :param volume: the desired volume.
    :param obj: the blender object to scale
    """
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    vol = bm.calc_volume()
    scaling_factor = (volume/vol)**(1/3)
    obj.scale = Vector([scaling_factor]*3)


def apply_surface_displacement():
    pass


def new_material(name: str) -> bpy.types.Material:
    """ A helper function for creating new materials in Blender.

    :param name: the name for the new material.
    """
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    if mat.node_tree:
        mat.node_tree.links.clear()
        mat.node_tree.nodes.clear()
    return mat


def add_surface_lighting(stl_file: str,
                         collection: bpy.types.Collection = None,
                         parent_object: bpy.types.Object = None,
                         scaling_factor: float = 1,
                         euler_rotation: List[float] = None,
                         emission_color: List[float] = None,
                         emission_strength: float = 100) \
        -> Tuple[bpy.types.Object, bpy.types.Node]:
    """
    Add an STL surface to the collection and make it light up.
    TODO: add ability to specify material settings

    :param stl_file: path to stl surface
    :param parent_object: optional parent object to assign this light to
    :param collection: the collection to link this object to. defaults to active one
    :param scaling_factor: uniform scaling factor to apply to the STL model
    :param euler_rotation: initial rotation to apply to STL model in degrees
    :param emission_strength: high bright the light is
    :param emission_color: the color of the light. defaults to [1, 1, 1, 1]
    :return: the added light object, the emission shader node
    """
    if euler_rotation is None:
        euler_rotation = [0, 0, 0]
    if emission_color is None:
        emission_color = [1, 1, 1, 1]
    bpy.ops.import_mesh.stl(filepath=stl_file)
    stl_object = bpy.data.objects[os.path.splitext(os.path.basename(stl_file))[0]]
    stl_object.scale = Vector([scaling_factor, scaling_factor, scaling_factor])
    stl_object.rotation_euler = Vector(np.radians(euler_rotation))
    if collection is not None:
        bpy.context.scene.collection.objects.unlink(stl_object)
        collection.objects.link(stl_object)

    if parent_object:
        stl_object.parent = parent_object
    mat = new_material('light_emission')
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    output = nodes.new(type='ShaderNodeOutputMaterial')
    shader = nodes.new(type='ShaderNodeEmission')
    mixer = nodes.new(type='ShaderNodeMixShader')
    backfacing = nodes.new(type='ShaderNodeNewGeometry')
    nodes["Emission"].inputs[0].default_value = emission_color
    nodes["Emission"].inputs[1].default_value = emission_strength
    links.new(backfacing.outputs[6], mixer.inputs[0])
    links.new(shader.outputs[0], mixer.inputs[1])
    links.new(mixer.outputs[0], output.inputs[0])
    stl_object.data.materials.append(mat)
    return stl_object, shader


def add_tumor_particle_nodegroup(stl_file: str,
                                 density: float = 10,
                                 volume_max: float = 0.1,
                                 scaling_range: Tuple[float] = None,
                                 rotation_range: Tuple[float] = None) \
        -> bpy.types.NodeGroup:
    """
    Scatter instances of the mesh in the stl-file over the targeted object.
    Note: Implemented with geometry nodes, probably easier and more readable with straight code.

    :param stl_file: path to the stl file to be used as tumor particle
    :param density: controls amount of of particles added
    :param volume_max: volume of object referenced for the instances
    :param scaling_range: range in which scaling of instances varies
    :param rotation_range: range in which rotation of instances varies
    :return: the modified object, the node creating points, the node creating instances
    """
    # create reference object from .stl-file
    bpy.ops.import_mesh.stl(filepath=stl_file)
    particle_ref_object = bpy.data.objects[os.path.splitext(os.path.basename(stl_file))[0]]
    scale_mesh_volume(particle_ref_object, volume_max)

    # set up node group
    particle_nodegroup = bpy.data.node_groups.new('particle-nodes', type='GeometryNodeTree')
    nodes = particle_nodegroup.nodes
    links = particle_nodegroup.links
    # create nodes
    group_in = nodes.new('NodeGroupInput')
    points_on_faces = nodes.new('GeometryNodeDistributePointsOnFaces')
    obj_info = nodes.new('GeometryNodeObjectInfo')
    instance_on_points = nodes.new('GeometryNodeInstanceOnPoints')
    rotation_vector = nodes.new('FunctionNodeRandomValue')
    scaling_int = nodes.new('FunctionNodeRandomValue')
    join_geo = nodes.new('GeometryNodeJoinGeometry')
    group_out = nodes.new('NodeGroupOutput')
    # set default parameters
    rotation_vector.data_type = 'FLOAT_VECTOR'
    rotation_vector.inputs.data.inputs['Min'].default_value = rotation_range[0]
    rotation_vector.inputs.data.inputs['Max'].default_value = rotation_range[1]
    scaling_int.inputs.data.inputs[2].default_value = scaling_range[0]
    scaling_int.inputs.data.inputs[3].default_value = scaling_range[1]
    points_on_faces.inputs['Density'].default_value = density
    obj_info.inputs['Object'].default_value = particle_ref_object
    # link nodes
    links.new(group_in.outputs['Geometry'], points_on_faces.inputs['Mesh'])
    links.new(group_in.outputs['Geometry'], join_geo.inputs['Geometry'])
    links.new(obj_info.outputs['Geometry'], instance_on_points.input['Instance'])
    links.new(points_on_faces.outputs['Points'], instance_on_points.input['Points'])
    links.new(rotation_vector.outputs['Value'], instance_on_points.input['Scale'])
    links.new(instance_on_points.outputs['Instances'], join_geo.inputs['Geometry'])
    links.new(join_geo.outputs['Geometry'], group_out['Geometry'])
    return particle_nodegroup


def add_render_output_nodes(scene: bpy.types.Scene,
                            color: bool = True,
                            depth: bool = True,
                            normals: bool = False,
                            view_layer: str = "ViewLayer") -> List[bpy.types.Node]:
    """
    Modify the graph of a scene's node tree to include color and depth outputs

    :param color: whether to include color as output.
    :param depth: whether to include the depth as output.
    :param normals: whether to include the normals as output.
    :param scene: the scene to create the rendering for.
    :param view_layer: the view layer in the scene to enable the rendering passes for.
    :returns: image node, depth node, normals node  <- will be None if option not enabled.
    """
    tree = scene.node_tree
    rl = tree.nodes.new('CompositorNodeRLayers')
    return_list = [None, None, None]
    links = tree.links
    if depth:
        scene.view_layers["ViewLayer"].use_pass_z = True
        # create depth output node
        depth_node = tree.nodes.new('CompositorNodeOutputFile')
        depth_node.format.file_format = "OPEN_EXR"
        links.new(rl.outputs['Depth'], depth_node.inputs['Image'])  # link Z to output
        return_list[1] = depth_node

    if color:
        # create image output node
        img_node = tree.nodes.new('CompositorNodeOutputFile')
        img_node.format.file_format = "PNG"
        links.new(rl.outputs['Image'], img_node.inputs['Image'])  # link image to output
        return_list[0] = img_node

    if normals:
        scene.view_layers["ViewLayer"].use_pass_normal = True
        # create normals output node
        normal_node = tree.nodes.new('CompositorNodeOutputFile')
        normal_node.format.file_format = "OPEN_EXR"
        links.new(rl.outputs['Normal'], normal_node.inputs['Image'])  # link Z to output
        return_list[2] = normal_node

    return return_list


def add_shrinkwrap_constraint(obj: bpy.types.Object,
                              config: ShrinkwrapConfig = ShrinkwrapConfig()) -> bpy.types.Constraint:
    """
    adds a shrinkwrap constraint to the provided object and sets the values provided in config.

    :param obj: a blender object
    :param config: the configuration for the shrinkwrap
    :return: the constraint handle
    """
    shrinkwrap_constr = obj.constraints.new("SHRINKWRAP")
    set_blender_data(shrinkwrap_constr, config)
    return shrinkwrap_constr


def set_gpu_rendering_preferences(gpu: int = -1) -> None:
    """
    Set GPU resources to use for rendering. This function only works for CUDA GPUs or Macs

    :param gpu: the GPU ID to use. if -1, uses all GPUs available.
    """
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'CUDA' if (sys.platform != 'darwin') else 'METAL'
    for dev in prefs.devices:
        if dev.type == "CPU":
            dev.use = True
    gpu_num = 0
    for dev in prefs.devices:
        if dev.type != "CPU":
            if gpu == -1:
                dev.use = True
            else:
                dev.use = gpu_num == gpu
            gpu_num += 1


def extract_system_arguments() -> Tuple[List[str], bool]:
    """
    Provides the user passed arguments that come after "--" when running a python script through blender.
    Also provides whether the script was called as headless

    :return: parsed_args, headless
    """
    idx = 0
    try:
        idx = sys.argv.index("--")
        cli_arguments = True
    except ValueError:
        cli_arguments = False
    arg_string = sys.argv[idx + 1:] if cli_arguments else ""

    gui_enabled = False
    try:
        gui_enabled = bool(sys.argv.index('-b'))
    except ValueError:
        pass

    return arg_string, not gui_enabled


