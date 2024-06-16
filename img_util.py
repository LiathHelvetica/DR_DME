import cv2
import numpy as np

from nptyping import NDArray, Shape

from constants import IMG_CROPPER_THRESHOLD


def img_to_square(path: str, threshold: int = IMG_CROPPER_THRESHOLD) -> NDArray[Shape["*, *, 3"], np.uint8]:
  img = cv2.imread(path)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  _, img_thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

  pts = np.argwhere(img_thresh > 0)
  y1, x1 = pts.min(axis=0)
  y2, x2 = pts.max(axis=0)

  img_out = img[y1:y2, x1:x2]
  size_out = max(y2 - y1, x2 - x1)

  orig_y_size = img_out.shape[0]
  orig_x_size = img_out.shape[1]
  px_pad_top = (size_out - orig_y_size) // 2
  px_pad_bot = (size_out - orig_y_size) - px_pad_top
  px_pad_left = (size_out - orig_x_size) // 2
  px_pad_right = (size_out - orig_x_size) - px_pad_left

  img_out = cv2.copyMakeBorder(
    img_out,
    px_pad_top,
    px_pad_bot,
    px_pad_left,
    px_pad_right,
    cv2.BORDER_CONSTANT,
    value = (0, 0, 0)
  )

  return img_out