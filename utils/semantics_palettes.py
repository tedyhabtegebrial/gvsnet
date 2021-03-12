from collections import namedtuple
import numpy as np

vkitti_pallete = [[0, 0, 0],
                  [210, 0, 200],
                  [90, 200, 255],
                  [0, 199, 0],
                  [90, 240, 0],
                  [140, 140, 140],
                  [100, 60, 100],
                  [250, 100, 255],
                  [255, 255, 0],
                  [200, 200, 0],
                  [255, 130, 0],
                  [80, 80, 80],
                  [160, 60, 60],
                  [255, 127, 80],
                  [0, 139, 139]]
vkitti_pallete = {k: vkitti_pallete[k] for k in range(len(vkitti_pallete))}

carla_pallete = {
    0: [0, 0, 0],  # None
    1: [70, 70, 70],  # Buildings
    2: [190, 153, 153],  # Fences
    3: [72, 0, 90],  # Other
    4: [220, 20, 60],  # Pedestrians
    5: [153, 153, 153],  # Poles
    6: [157, 234, 50],  # RoadLines
    7: [128, 64, 128],  # Roads
    8: [244, 35, 232],  # Sidewalks
    9: [107, 142, 35],  # Vegetation
    10: [0, 0, 255],  # Vehicles
    11: [102, 102, 156],  # Walls
    12: [220, 220, 0],  # TrafficSigns
    13: [150, 33, 88],  # TrafficSigns
    14: [111, 74,  0],
    15: [81, 0, 81],
    16: [250, 170, 160],
    17: [230, 150, 140],
    18: [180, 165, 180],
    19: [150, 100, 100],
    20: [150, 120, 90],
    21: [250, 170, 30],
    22: [220, 220,  0],
    23: [152, 251, 152],
    24: [70, 130, 180],
    25: [255, 0, 0],
    26: [0, 0, 142],
    27: [0, 0, 70],
    28: [0, 60, 100],
    29: [0, 0, 110],
    20: [0, 80, 100],
    31: [0, 0, 230],
    32: [119, 11, 32],
}


CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                 'has_instances', 'ignore_in_eval', 'color'])
classes = [
    CityscapesClass('unlabeled',            0, 255,
                    'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('ego vehicle',          1, 255,
                    'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('rectification border', 2, 255,
                    'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('out of roi',           3, 255,
                    'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('static',               4, 255,
                    'void', 0, False, True, (0, 0, 0)),
    CityscapesClass('dynamic',              5, 255,
                    'void', 0, False, True, (111, 74, 0)),
    CityscapesClass('ground',               6, 255,
                    'void', 0, False, True, (81, 0, 81)),
    CityscapesClass('road',                 7, 0, 'flat',
                    1, False, False, (128, 64, 128)),
    CityscapesClass('sidewalk',             8, 1, 'flat',
                    1, False, False, (244, 35, 232)),
    CityscapesClass('parking',              9, 255, 'flat',
                    1, False, True, (250, 170, 160)),
    CityscapesClass('rail track',           10, 255, 'flat',
                    1, False, True, (230, 150, 140)),
    CityscapesClass('building',             11, 2,
                    'construction', 2, False, False, (70, 70, 70)),
    CityscapesClass('wall',                 12, 3,
                    'construction', 2, False, False, (102, 102, 156)),
    CityscapesClass('fence',                13, 4,
                    'construction', 2, False, False, (190, 153, 153)),
    CityscapesClass('guard rail',           14, 255,
                    'construction', 2, False, True, (180, 165, 180)),
    CityscapesClass('bridge',               15, 255,
                    'construction', 2, False, True, (150, 100, 100)),
    CityscapesClass('tunnel',               16, 255,
                    'construction', 2, False, True, (150, 120, 90)),
    CityscapesClass('pole',                 17, 5, 'object',
                    3, False, False, (153, 153, 153)),
    CityscapesClass('polegroup',            18, 255, 'object',
                    3, False, True, (153, 153, 153)),
    CityscapesClass('traffic light',        19, 6, 'object',
                    3, False, False, (250, 170, 30)),
    CityscapesClass('traffic sign',         20, 7, 'object',
                    3, False, False, (220, 220, 0)),
    CityscapesClass('vegetation',           21, 8, 'nature',
                    4, False, False, (107, 142, 35)),
    CityscapesClass('terrain',              22, 9, 'nature',
                    4, False, False, (152, 251, 152)),
    CityscapesClass('sky',                  23, 10, 'sky',
                    5, False, False, (70, 130, 180)),
    CityscapesClass('person',               24, 11,
                    'human', 6, True, False, (220, 20, 60)),
    CityscapesClass('rider',                25, 12,
                    'human', 6, True, False, (255, 0, 0)),
    CityscapesClass('car',                  26, 13,
                    'vehicle', 7, True, False, (0, 0, 142)),
    CityscapesClass('truck',                27, 14,
                    'vehicle', 7, True, False, (0, 0, 70)),
    CityscapesClass('bus',                  28, 15,
                    'vehicle', 7, True, False, (0, 60, 100)),
    CityscapesClass('caravan',              29, 255,
                    'vehicle', 7, True, True, (0, 0, 90)),
    CityscapesClass('trailer',              30, 255,
                    'vehicle', 7, True, True, (0, 0, 110)),
    CityscapesClass('train',                31, 16,
                    'vehicle', 7, True, False, (0, 80, 100)),
    CityscapesClass('motorcycle',           32, 17,
                    'vehicle', 7, True, False, (0, 0, 230)),
    CityscapesClass('bicycle',              33, 18,
                    'vehicle', 7, True, False, (119, 11, 32)),
    CityscapesClass('license plate',        -1, 255,
                    'vehicle', 7, False, True, (0, 0, 142)),
]

train_id_to_color = [c.color for c in classes if (
    c.train_id != -1 and c.train_id != 255)]
train_id_to_color.append([0, 0, 0])
train_id_to_color = np.array(train_id_to_color)

cityscapes_pallete = {i: v for i, v in enumerate(train_id_to_color)}
scene_net_pallete = {
    0: [0, 0, 0],
    1: [0, 0, 255],
    2: [232, 88, 47],
    3: [0, 217, 0],
    4: [148, 0, 240],
    5: [222, 241, 23],
    6: [255, 205, 205],
    7: [0, 223, 228],
    8: [106, 135, 204],
    9: [116, 28, 41],
    10: [240, 35, 235],
    11: [0, 166, 156],
    12: [249, 139, 0],
    13: [225, 228, 194],
}


scan_net_pallete = {
    0: [128, 64, 128],
    1: [244, 35, 232],
    2: [70, 70, 70],
    3: [102, 102, 156],
    4: [190, 153, 153],
    5: [153, 153, 153],
    6: [250, 170, 30],
    7: [220, 220, 0],
    8: [107, 142, 35],
    9: [152, 251, 152],
    10: [70, 130, 180],
    11: [220, 20, 60],
    12: [255, 0, 0],
    13: [0, 0, 142],
    14: [0, 0, 70],
    15: [0, 60, 100],
    6: [0, 80, 100],
    17: [0, 0, 230],
    18: [119, 11, 32],
    19: [0, 0, 230],
}


def get_palette(dataset_name):
    if dataset_name.startswith('carla_'):
        dataset_name = 'carla'
    pallet_map = {}
    pallet_map['carla'] = carla_pallete
    pallet_map['vkitti'] = vkitti_pallete
    pallet_map['cityscapes'] = cityscapes_pallete
    assert dataset_name in pallet_map.keys(
    ), f'Unknown dataset {dataset_name}: not in {pallet_map.keys()} '
    return pallet_map[dataset_name]


def get_num_classes(dataset_name):
    if dataset_name.startswith('carla_'):
        dataset_name = 'carla'
    num_classes_map = {}
    num_classes_map['carla'] = 13
    # num_classes_map['scan_net'] = 13
    # num_classes_map['scenenet_rgbd'] = 14
    num_classes_map['vkitti'] = 16
    num_classes_map['cityscapes'] = 20
    assert dataset_name in num_classes_map.keys(
    ), f'Unknown dataset {dataset_name}: not in {num_classes_map.keys()} '
    return num_classes_map[dataset_name]
