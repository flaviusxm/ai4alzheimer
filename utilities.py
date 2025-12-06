import cv2 as cv
import os


def show_image(img, title):
    img = cv.resize(img, (0, 0), fx=0.3, fy=0.3)
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def save_image(img, path, file_name):
    if path is None:
        return

    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory {path}")

    full_path = f"{path}/{file_name}"
    success = cv.imwrite(full_path, img)

    if not success:
        print(f"Failed to save {full_path}")
