import cv2
import os
import glob
import json
import shutil
import numpy as np


def save_labelme_format(image_path, contour, output_dir, label):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    shapes = []

    points = contour.reshape(-1, 2).tolist()
    shape = {
        "label": label,
        "points": points,
        "group_id": None,
        "shape_type": "polygon",
        "flags": {}
    }
    shapes.append(shape)

    labelme_data = {
        "version": "4.5.6",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w
    }

    json_name = os.path.splitext(os.path.basename(image_path))[0] + '.json'
    with open(os.path.join(output_dir, json_name), 'w') as f:
        json.dump(labelme_data, f, indent=4)


def is_valid_contour(contour, image_shape, min_area_ratio=0.01):
    img_h, img_w = image_shape[:2]
    min_area = img_h * img_w * min_area_ratio
    return cv2.contourArea(contour) > min_area


def augment_image(image):
    rows, cols = image.shape[:2]

    # 随机旋转
    angle = np.random.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, M, (cols, rows), borderValue=(255, 255, 255))

    # 随机仰角变化
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[50, 50], [200, 50 + np.random.uniform(-50, 50)], [50, 200 + np.random.uniform(-50, 50)]])
    M = cv2.getAffineTransform(pts1, pts2)
    skewed_image = cv2.warpAffine(rotated_image, M, (cols, rows), borderValue=(255, 255, 255))

    return skewed_image


def find_black_numbers(image):
    # 转换为灰度图像并进行二值化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # 寻找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤面积小于最小面积的轮廓
    img_h, img_w = image.shape[:2]
    min_area = img_h * img_w * 0.01
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    # 进一步过滤，确保轮廓是黑色数字并被白色背景包围
    final_contours = []
    for contour in valid_contours:
        x, y, w, h = cv2.boundingRect(contour)
        if x > 0 and y > 0 and (x + w) < img_w and (y + h) < img_h:
            if np.all(image[y-1:y+h+1, x-1:x+w+1] == 255):
                final_contours.append(contour)

    return final_contours


def process_images_and_merge(folders, output_dir, num_augmented=5):
    os.makedirs(output_dir, exist_ok=True)

    for label, folder in enumerate(folders, start=1):
        image_paths = glob.glob(os.path.join(folder, '*.jpg'))
        print(f"Processing folder: {folder}, found {len(image_paths)} images.")
        for image_path in image_paths:
            print(f"Processing image: {image_path}")
            img = cv2.imread(image_path)
            valid_contours = find_black_numbers(img)

            if valid_contours:
                largest_contour = max(valid_contours, key=cv2.contourArea)

                # 原图和标注保存
                shutil.copy(image_path, output_dir)
                output_image_path = os.path.join(output_dir, os.path.basename(image_path))
                save_labelme_format(output_image_path, largest_contour, output_dir, str(label))

                # 数据增强和保存
                for i in range(num_augmented):
                    augmented_image = augment_image(img)
                    aug_image_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_aug_{i}.jpg"
                    aug_image_path = os.path.join(output_dir, aug_image_name)
                    cv2.imwrite(aug_image_path, augmented_image)

                    # 对增强后的图像重新计算轮廓
                    aug_valid_contours = find_black_numbers(augmented_image)

                    if aug_valid_contours:
                        aug_largest_contour = max(aug_valid_contours, key=cv2.contourArea)
                        save_labelme_format(aug_image_path, aug_largest_contour, output_dir, str(label))

    print("Batch annotation and merging completed.")


if __name__ == '__main__':
    folders = [
        # 这里是需要遍历的文件夹目录
        "C:/Users/21093/Desktop/YOLOv5-Lite-1.4/photo/images/1",
        "C:/Users/21093/Desktop/YOLOv5-Lite-1.4/photo/images/2",
        "C:/Users/21093/Desktop/YOLOv5-Lite-1.4/photo/images/3",
        "C:/Users/21093/Desktop/YOLOv5-Lite-1.4/photo/images/4",
        "C:/Users/21093/Desktop/YOLOv5-Lite-1.4/photo/images/5",
        "C:/Users/21093/Desktop/YOLOv5-Lite-1.4/photo/images/6",
        "C:/Users/21093/Desktop/YOLOv5-Lite-1.4/photo/images/7",
        "C:/Users/21093/Desktop/YOLOv5-Lite-1.4/photo/images/8"
    ]
    # 这里是输出目录
    output_dir = "C:/Users/21093/Desktop/YOLOv5-Lite-1.4/photo/other/images1"
    process_images_and_merge(folders, output_dir)