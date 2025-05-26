import cv2
import numpy as np

def margin_crop_images(uploaded_files):
    """
    Perform Sobel vertical margin crop on each uploaded file.
    Returns a dict mapping file_name to (crop_img, margin_x, margin_sum).
    """
    margin_crops = {}
    for file in uploaded_files:
        # Ensure reading from start and reset pointer after to allow re-reads
        file.seek(0)
        content = file.read()
        file.seek(0)
        arr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        _, binary = cv2.threshold(scaled_sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(5, h // 50)))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        left_region = closed[:, :w // 2]
        col_sums = np.sum(left_region, axis=0)
        margin_x = int(np.argmax(col_sums))
        margin_sum = col_sums[margin_x]
        crop_img = img[:, :margin_x]
        margin_crops[file.name] = (crop_img, margin_x, margin_sum)
    return margin_crops 