import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import cv2


def colorFader(c1, c2, mix):
    mix = 0 if mix < 0 else (1 if 1 < mix else mix)
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    c = ((1 - mix) * c1 + mix * c2) * 255
    rgb = c.astype(np.uint8)
    return rgb_to_bgr(rgb)


def makeColorFunc(c1, c2):
    return lambda mix: colorFader(c1, c2, mix)


def makeColorFunc2(c1, c2, c3):
    return lambda mix: colorFader(c1, c2, mix * 2) if mix < 0.5 else colorFader(c2, c3, (mix - 0.5) * 2)


DEFAULT_COLOR_FUNC = makeColorFunc("#FFFFFF", "#006400")
DEFAULT_COLOR_FUNC2 = makeColorFunc2("#0000ff", "#FFFFFF", "#FF0000")


def makeColorMap(matrix, cell_x, cell_y,
                 color_func=DEFAULT_COLOR_FUNC, boundary=None, axis=None,
                 border_mask=None, border_color="#dc143c", border_size=2):
    """
    color_func: (float)[0, 1] -> color
    boundary: [lower, upper] or None
    axis: axis to apply color gradation (None or 0 or 1)
          (ignored if boundary is specified by numeric values)
    border_mask: bool matrix (the same shape as param matrix)
    """
    mat = np.array(matrix, dtype=np.float64)

    if(boundary is None or boundary[0] is None):
        lower = mat.min(axis=axis, keepdims=True)
    elif(isinstance(boundary[0], (int, float))):
        lower = np.full_like(mat, boundary[0])
    else:
        lower = boundary[0]

    if(boundary is None or boundary[1] is None):
        upper = mat.max(axis=axis, keepdims=True)
    elif(isinstance(boundary[1], (int, float))):
        upper = np.full_like(mat, boundary[1])
    else:
        upper = boundary[1]
    mat = (mat - lower) / (upper - lower)

    img = np.zeros((mat.shape[0], mat.shape[1], 3))
    # img = color_func(mat)
    for x in range(mat.shape[0]):
        for y in range(mat.shape[1]):
            img[x, y, :] = color_func(mat[x, y])

    img = np.repeat(img, cell_x, axis=0)
    img = np.repeat(img, cell_y, axis=1)

    if(border_mask is not None):
        border_color = to_bgr(border_color)
        border = np.zeros((cell_x, cell_y, 3), dtype=np.uint8)
        border[:, :, 0] = border_color[0]
        border[:, :, 1] = border_color[1]
        border[:, :, 2] = border_color[2]
        mask = np.full((cell_x, cell_y), True)
        mask[border_size:-border_size - 1, border_size:-border_size - 1] = False

        for x in range(mat.shape[0]):
            for y in range(mat.shape[1]):
                if(border_mask[x, y]):
                    addImage(img, border, x * cell_x, y * cell_y, mask)

    return img


def addImage(img1, img2, x, y, mask=None):
    """
    Add img2 on img1
    """
    h = img2.shape[0] if img1.shape[0] > x + img2.shape[0] else img1.shape[0] - x
    w = img2.shape[1] if img1.shape[1] > y + img2.shape[1] else img1.shape[1] - y
    if(mask is None):
        img1[x:x + h, y:y + w, :] = img2[0:h, 0:w, :]
    else:
        img1[x:x + h, y:y + w, :][mask[0:h, 0:w]] = img2[0:h, 0:w, :][mask[0:h, 0:w]]

    return img1


def to_bgr(color):
    rgb = mpl.colors.to_rgb(color)
    rgb = (np.array(rgb) * 255).astype(np.uint8)
    return rgb_to_bgr(rgb)


def rgb_to_bgr(rgb):
    bgr = rgb.copy()
    bgr[2] = rgb[0]
    bgr[0] = rgb[2]
    return bgr
