import os
import json
import numpy as np
import cv2
import re
import locale
import natsort
import matplotlib.pyplot as plt
from PIL import Image
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, load_image, predict, annotate, bbox_intersection

HOME = "/home/jykim/JY_project/Few_Shot/"
CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")  # set model configuration file path
WEIGHTS_NAME = "groundingdino_swint_ogc.pth"  # set model weight file ath
WEIGHTS_PATH = os.path.join(HOME, "GroundingDINO/weights", WEIGHTS_NAME)
model = load_model(CONFIG_PATH, WEIGHTS_PATH)
xyxy = []

def inference(img, prompt, box_threshold=0.35, text_threshold=0.25):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_transformed, _ = transform(Image.fromarray(img), None)

    boxes, logits, phrases = predict(
        model=model,
        image=image_transformed,
        caption=prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
    bbox_xyxy = bbox_intersection(image_source=img, boxes=boxes)
    numpy2List = bbox_xyxy.tolist()
    xyxy.append(numpy2List)
    annotated_frame = annotate(image_source=img, boxes=boxes, logits=logits, phrases=phrases)

    return annotated_frame

def getpreferredencoding(do_setlocale = True):
    return "UTF-8"

locale.getpreferredencoding = getpreferredencoding


# image caption 
file_path = "/home/jykim/JY_project/Few_Shot/BLIP/annotation/density/new_density.json"

with open(file_path, 'r') as f:
    fall_json = json.load(f)

# image load
path = "./dataset/density/"
paths = [os.path.join(path , i) for i in os.listdir(path) if re.search(".jpg$", i )]
paths = natsort.natsorted(paths)
people = []

# # ------------------
# grounding dino HOI
for idx, img_path in enumerate(paths):
    people_cnt = 0  
    img = cv2.imread(paths[idx])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    caption = fall_json[idx][1]

    # fall detection(BLIP + DINO + HOI)
    if "man" in caption or "men" in caption or "people" in caption or "crowded" in caption or "group" in caption:
        TEXT_PROMPT1 = "man"

        try:
            print("man")
            result_img1= inference(img, TEXT_PROMPT1)
            result_img1 = cv2.cvtColor(result_img1, cv2.COLOR_BGR2RGB) 
            result_img1 = result_img1[:, :, ::-1]
            people_cnt = len(xyxy)
            people.append(people_cnt)
        except:
            result_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        print("no man")
        result_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 8))
    plt.imshow(result_img1[:, :, ::-1])
    plt.axis("off")
    if idx < 9:
        save_img = "./result/density/density00" + str(idx+1) + ".png"
    elif 8 < idx < 99:
        save_img = "./result/density/density0" + str(idx+1) + ".png"
    else:
        save_img = "./result/density/density" + str(idx+1) + ".png"  

    plt.savefig(save_img)
    xyxy = []

print("people count : {}".format(people))
print("done.")




