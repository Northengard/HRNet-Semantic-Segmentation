from numpy import zeros, argmax, asarray
from cv2 import resize, addWeighted

COLORS = {
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
    16: [0, 80, 100],
    17: [0, 0, 230],
    18: [119, 11, 32],
    19: [81, 0, 81]
}


def get_colored_frame(frame, predicted, overlay_coef=0.5):
    global COLORS
    blank_img = zeros(frame.shape, dtype='uint8')
    output = predicted.cpu().numpy().squeeze().transpose(1, 2, 0)
    output = resize(output, frame.shape[:2][::-1])
    output = asarray(argmax(output, axis=2), dtype='uint8')
    for cls in COLORS.keys():
        mask = output == cls
        blank_img[mask > 0.5] = COLORS[cls]
    predicted = addWeighted(frame, (1 - overlay_coef), blank_img, overlay_coef, 0)
    return predicted