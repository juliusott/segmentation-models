import numpy as np
from collections import namedtuple
import matplotlib.pyplot as plt

# a label and all meta information
# Code inspired by Cityscapes https://github.com/mcordts/cityscapesScripts
Label = namedtuple('Label', [

    'name',  # The identifier of this label, e.g. 'car', 'person', ... .
    # We use them to uniquely name a class

    'id',  # An integer ID that is associated with this label.
    # The IDs are used to represent the label in ground truth images
    # An ID of -1 means that this label does not have an ID and thus
    # is ignored when creating ground truth images (e.g. license plate).
    # Do not modify these IDs, since exactly these IDs are expected by the
    # evaluation server.

    'trainId',
    # Feel free to modify these IDs as suitable for your method. Then create
    # ground truth images with train IDs, using the tools provided in the
    # 'preparation' folder. However, make sure to validate or submit results
    # to our evaluation server using the regular IDs above!
    # For trainIds, multiple labels might have the same ID. Then, these labels
    # are mapped to the same class in the ground truth images. For the inverse
    # mapping, we use the label that is defined first in the list below.
    # For example, mapping all void-type classes to the same ID in training,
    # might make sense for some approaches.
    # Max value is 255!

    'category',  # The name of the category that this label belongs to

    'categoryId',
    # The ID of this category. Used to create ground truth images
    # on category level.

    'hasInstances',
    # Whether this label distinguishes between single instances or not

    'ignoreInEval',
    # Whether pixels having this class as ground truth label are ignored
    # during evaluations or not

    'color',  # The color of this label
])
labels = [
    #       name                     id     trainId category            catId   hasInstances   ignoreInEval     color
    Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('dynamic', 1, 255, 'void', 0, False, True, (111, 74, 0)),
    Label('ego vehicle', 2, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('ground', 3, 255, 'void', 0, False, True, (81, 0, 81)),
    Label('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
    Label('parking', 5, 255, 'flat', 1, False, True, (250, 170, 160)),
    Label('rail track', 6, 255, 'flat', 1, False, True, (230, 150, 140)),
    Label('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
    Label('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
    Label('bridge', 9, 255, 'construction', 2, False, True, (150, 100, 100)),
    Label('building', 10, 2, 'construction', 2, False, False, (70, 70, 70)),
    Label('fence', 11, 4, 'construction', 2, False, False, (190, 153, 153)),
    Label('garage', 12, 255, 'construction', 2, False, True, (180, 100, 180)),
    Label('guard rail', 13, 255, 'construction', 2, False, True, (180, 165, 180)),
    Label('tunnel', 14, 255, 'construction', 2, False, True, (150, 120, 90)),
    Label('wall', 15, 3, 'construction', 2, False, False, (102, 102, 156)),
    Label('banner', 16, 255, 'object', 3, False, True, (250, 170, 100)),
    Label('billboard', 17, 255, 'object', 3, False, True, (220, 220, 250)),
    Label('lane divider', 18, 255, 'object', 3, False, True, (255, 165, 0)),
    Label('parking sign', 19, 255, 'object', 3, False, False, (220, 20, 60)),
    Label('pole', 20, 5, 'object', 3, False, False, (153, 153, 153)),
    Label('polegroup', 21, 255, 'object', 3, False, True, (153, 153, 153)),
    Label('street light', 22, 255, 'object', 3, False, True, (220, 220, 100)),
    Label('traffic cone', 23, 255, 'object', 3, False, True, (255, 70, 0)),
    Label('traffic device', 24, 255, 'object', 3, False, True, (220, 220, 220)),
    Label('traffic light', 25, 6, 'object', 3, False, False, (250, 170, 30)),
    Label('traffic sign', 26, 7, 'object', 3, False, False, (220, 220, 0)),
    Label('traffic sign frame', 27, 255, 'object', 3, False, True, (250, 170, 250)),
    Label('terrain', 28, 9, 'nature', 4, False, False, (152, 251, 152)),
    Label('vegetation', 29, 8, 'nature', 4, False, False, (107, 142, 35)),
    Label('sky', 30, 10, 'sky', 5, False, False, (70, 130, 180)),
    Label('person', 31, 11, 'human', 6, True, False, (220, 20, 60)),
    Label('rider', 32, 12, 'human', 6, True, False, (255, 0, 0)),
    Label('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
    Label('bus', 34, 15, 'vehicle', 7, True, False, (0, 60, 100)),
    Label('car', 35, 13, 'vehicle', 7, True, False, (0, 0, 142)),
    Label('caravan', 36, 255, 'vehicle', 7, True, True, (0, 0, 90)),
    Label('motorcycle', 37, 17, 'vehicle', 7, True, False, (0, 0, 230)),
    Label('trailer', 38, 255, 'vehicle', 7, True, True, (0, 0, 110)),
    Label('train', 39, 16, 'vehicle', 7, True, False, (0, 80, 100)),
    Label('truck', 40, 14, 'vehicle', 7, True, False, (0, 0, 70)),
]

categories = {"void": [(0, 0, 0), (111, 74, 0), (81, 0, 81)],
              "flat": [(250, 170, 160), (230, 150, 140), (128, 64, 128)],
              "construction": [(150, 100, 100), (70, 70, 70), (190, 153, 153), (180, 100, 180), (180, 165, 180),
                               (150, 120, 90), (102, 102, 156)],
              "object": [(250, 170, 100), (220, 220, 250), (255, 165, 0), (220, 20, 60), (153, 153, 153),
                         (220, 220, 100), (255, 70, 0), (220, 220, 220), (250, 170, 30), (220, 220, 0), (250, 170, 250),
                         (244, 35, 232)],
              "nature": [(152, 251, 152), (107, 142, 35)],
              "sky": [(70, 130, 180)],
              "human": [(220, 20, 60), (255, 0, 0)],
              "vehicle": [(119, 11, 32), (0, 60, 100), (0, 0, 142), (0, 0, 90), (0, 0, 230), (0, 0, 110), (0, 80, 100),
                          (0, 0, 70)]
              }

SEG_LABELS_LIST = [
    {"id": -1, "name": "void", "rgb_values": [0, 0, 0]},
    {"id": 0, "name": "flat", "rgb_values": [0, 128, 0]},
    {"id": 1, "name": "construction", "rgb_values": [128, 128, 0]},
    {"id": 2, "name": "object", "rgb_values": [0, 0, 128]},
    {"id": 3, "name": "nature", "rgb_values": [128, 0, 128]},
    {"id": 4, "name": "sky", "rgb_values": [0, 128, 128]},
    {"id": 5, "name": "human", "rgb_values": [128, 128, 128]},
    {"id": 6, "name": "vehicle", "rgb_values": [64, 0, 0]}]


def label_img_to_rgb(label_img):
    label_img = np.squeeze(label_img)
    labels = np.unique(label_img)
    label_infos = [l for l in SEG_LABELS_LIST if l['id'] in labels]
    print(label_img.shape)
    label_img_rgb = np.array([label_img,
                              label_img,
                              label_img]).transpose(1, 2, 0)
    for l in label_infos:
        mask = label_img == l['id']
        label_img_rgb[mask] = l['rgb_values']

    return label_img_rgb.astype(np.uint8)


def create_gt(category, class_number, image):
    """
    gt = ground truth
    """
    image = image * 255
    image = image.astype(int)
    if category not in categories.keys():
        print("category not known")
    colors = categories[category]
    gt_init = np.zeros((image.shape[0], image.shape[1]))
    gt = gt_init
    for color in colors:
        try:
            gt_next = np.all(image == color, axis=2)
            gt = np.logical_or(gt, gt_next)
        except Exception :
            gt = np.logical_or(gt, gt_init)
    return gt * class_number


def image2label(seg_image):
    target_shape = (seg_image.shape[0], seg_image.shape[1])
    gt_0 = create_gt("void", 1, seg_image)
    gt_1 = create_gt("flat", 2, seg_image)
    gt_2 = create_gt("construction", 3, seg_image)
    gt_3 = create_gt("object", 4, seg_image)
    gt_4 = create_gt("nature", 5, seg_image)
    gt_5 = create_gt("sky", 6, seg_image)
    gt_6 = create_gt("human", 7, seg_image)
    gt_7 = create_gt("vehicle", 8, seg_image)
    # reshape array for concatenation
    # gt_0 = gt_0.reshape(*gt_0.shape, 1)
    # gt_1 = gt_1.reshape(*gt_1.shape, 1)
    gt_1 = np.add(gt_0, gt_1)

    # gt_2 = gt_2.reshape(*gt_2.shape, 1)
    gt_2 = np.add(gt_2, gt_1)

    # gt_3 = gt_3.reshape(*gt_3.shape, 1)
    gt_3 = np.add(gt_3, gt_2)

    # gt_4 = gt_4.reshape(*gt_4.shape, 1)
    gt_4 = np.add(gt_3, gt_4)

    # gt_5 = gt_5.reshape(*gt_5.shape, 1)
    gt_5 = np.add(gt_5, gt_4)

    # gt_6 = gt_6.reshape(*gt_6.shape, 1)
    gt_6 = np.add(gt_5, gt_6)

    # gt_7 = gt_7.reshape(*gt_7.shape, 1)
    gt_7 = np.add(gt_7, gt_6)

    gt_image = gt_7.reshape(target_shape)
    gt_image = np.subtract(gt_image, 1)
    gt_image[gt_image >= 8] = -1
    return np.array(gt_image)


def visualize_data(dataset):
    rgb = dataset[0][0]
    label_img = dataset[0][1]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(rgb.numpy().transpose(1, 2, 0))
    ax1.set_title("input image")
    ax2.imshow(label_img_to_rgb(label_img.numpy()))
    ax2.set_title("target image")
    plt.show()
