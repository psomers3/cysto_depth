import os
from typing import *
from pathlib import Path
from utils.exr_utils import exr_2_numpy
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
import cv2
import numpy as np
=======
>>>>>>> fixed camera clipping and fooling around with the Neural Network stuff
=======
import cv2
import numpy as np
>>>>>>> various changes:
=======
import cv2
import numpy as np
>>>>>>> e59494ca88f41def125ec679eee88acba6f7b805


def delete_clipping(depth_path: str, associated_paths: Union[List[str], str]):
    """given folder with .exr files containing depth information identify files with clipping and delete corresponding
    files in all passed paths"""
    bad_files = []
    for f in os.listdir(depth_path):
        depth_map = exr_2_numpy(os.path.join(depth_path, f))
        if depth_map.max() >= 1000:
            split_path = os.path.splitext(f)
            bad_files.append(split_path[0])

    associated_paths = associated_paths if isinstance(associated_paths, List) else [associated_paths]
    associated_paths.append(depth_path)

    for path in associated_paths:
        # assuming all files in one folder have the same suffix
        list_dir = os.listdir(path)
        suffix = Path(list_dir[0]).suffix
        for file in bad_files:
            os.remove(os.path.join(path, file) + suffix)
            #os.remove(files[bad_idx])


def reorganize(origin, destination):
    """one time function to create a reordered copy of a datatree via symbolic links"""
    list = Path(origin).glob('*')
    try:
        Path(destination,'color').mkdir(parents=True)
        Path(destination, 'depth').mkdir()
    except:
        print('color and depth folder already there or issue with mkdir')
    for f in Path(origin).glob('*'):
        print(str(f.name))
        if Path(origin, f).is_dir():
            [Path(Path(destination).resolve(), 'color', f.name + '_' + file.name).symlink_to(file) for file in
             Path(origin, f).rglob('*.png')]
            [Path(Path(destination).resolve(), 'depth', f.name + '_' + file.name).symlink_to(file) for file in
                Path(origin, f).rglob('*.exr')]


<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> various changes:
=======
>>>>>>> e59494ca88f41def125ec679eee88acba6f7b805
def remove_salt_noise(path):
    for image_path in Path(path).glob('*'):
        print(str(image_path))
        image = cv2.imread(str(image_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        median_image = cv2.medianBlur(image, ksize=3)
        cv2.imwrite(str(image_path), median_image)

<<<<<<< HEAD
<<<<<<< HEAD
if __name__ == '__main__':
  #  reorganize('/scratch/datasets/cysto_depth/depth_data_BlankMaterial_BlankMaterial_par_dis', '../../datasets/Johannes/BlankMaterials_par_dis')
  #  delete_clipping('../../datasets/Johannes/BlankMaterials_par_dis/depth',    '../../datasets/Johannes/BlankMaterials_par_dis/color')
    remove_salt_noise('../../datasets/Particles_diverticulum_tool_materials_151222/depth/bladder_wall')
=======
if __name__ == '__main__':
  #  reorganize('/scratch/datasets/cysto_depth/depth_data_BlankMaterial_BlankMaterial_par_dis', '../../datasets/Johannes/BlankMaterials_par_dis')
    delete_clipping('../../datasets/Johannes/BlankMaterials_par_dis/depth',    '../../datasets/Johannes/BlankMaterials_par_dis/color')
>>>>>>> fixed camera clipping and fooling around with the Neural Network stuff
=======
if __name__ == '__main__':
  #  reorganize('/scratch/datasets/cysto_depth/depth_data_BlankMaterial_BlankMaterial_par_dis', '../../datasets/Johannes/BlankMaterials_par_dis')
  #  delete_clipping('../../datasets/Johannes/BlankMaterials_par_dis/depth',    '../../datasets/Johannes/BlankMaterials_par_dis/color')
    remove_salt_noise('../../datasets/Particles_diverticulum_tool_materials_151222/depth/bladder_wall')
>>>>>>> various changes:
=======
if __name__ == '__main__':
  #  reorganize('/scratch/datasets/cysto_depth/depth_data_BlankMaterial_BlankMaterial_par_dis', '../../datasets/Johannes/BlankMaterials_par_dis')
  #  delete_clipping('../../datasets/Johannes/BlankMaterials_par_dis/depth',    '../../datasets/Johannes/BlankMaterials_par_dis/color')
    remove_salt_noise('../../datasets/Particles_diverticulum_tool_materials_151222/depth/bladder_wall')
>>>>>>> e59494ca88f41def125ec679eee88acba6f7b805
