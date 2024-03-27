from ultralytics import YOLO
import cv2
import numpy as np


def test_the_model():
    
    model = YOLO(
        r"models/best.pt"
    ) # This is from above we will create best.pt and use here.
    image = cv2.imread(
        r"D:\AMAR_Gitwork\yolov8_instance_segmentation\Copy-of-image8_png_jpg.rf.f7bab6581f69f33bf56a25c57927a652.jpg",
        1,
    )
    results = model.predict(
        image,
        retina_masks=True,
    )
    print(results[0].masks)

    if results[0].masks == None:
        print("No Mask Found")
        return  None
    new = None
    for result in results:
        mask = result.masks.cpu().numpy()
        bbox = result.boxes[0].cpu().numpy()
        masks = mask.data.astype(bool)
        ori_img = result.orig_img
        new = np.zeros_like(ori_img, dtype=np.uint8)
        for m in masks:
            
            new[m] = ori_img[m]

    cv2.imshow("image",new)
    cv2.waitKey(0)

    cv2.imwrite(
        "test.png",new)

test_the_model()