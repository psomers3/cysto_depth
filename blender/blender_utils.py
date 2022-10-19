import numpy as np
import os
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


def add_render_output_nodes(scene: bpy.types.Scene) -> Tuple[bpy.types.Node, bpy.types.Node]:
    """
    Modify the graph of a scene's node tree to include color and depth outputs
    :param scene: the scene to create the rending for
    :returns: depth node, image node
    """
    tree = scene.node_tree
    rl = tree.nodes.new('CompositorNodeRLayers')
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True  # TODO: make this non-context based

    # create depth output node
    depth_node = tree.nodes.new('CompositorNodeOutputFile')
    depth_node.format.file_format = "OPEN_EXR"

    # create image output node
    img_node = tree.nodes.new('CompositorNodeOutputFile')
    img_node.format.file_format = "PNG"

    # Links
    links = tree.links
    links.new(rl.outputs[2], depth_node.inputs['Image'])  # link Z to output
    links.new(rl.outputs['Image'], img_node.inputs['Image'])  # link image to output
    return depth_node, img_node


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


