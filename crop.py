import json
import os
import cv2

# data path
image_dir = 'train/image'
anno_dir = 'train/annos'
output_dir = 'stage2_crops'
empty = [0, 0, 0, 0]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for json_file in os.listdir(anno_dir):
    if not json_file.endswith('.json'):
        continue
        
    # Read the json
    with open(os.path.join(anno_dir, json_file), 'r') as f:
        data = json.load(f)
    
    # Load the corresponding image
    img_name = json_file.replace('.json', '.jpg')
    image = cv2.imread(os.path.join(image_dir, img_name))
    
    if image is None: continue

    # Read through the items
    for key in data.keys():
        if 'item' in key:
            item = data[key]
            bbox = item['bounding_box']     # [x1, y1, x2, y2]

            if (bbox == empty):
                continue
            
            x1, y1, x2, y2 = bbox
            
            crop_img = image[int(y1):int(y2), int(x1):int(x2)]
            
            crop_name = f"{img_name.split('.')[0]}_{key}.jpg"
            cv2.imwrite(os.path.join(output_dir, crop_name), crop_img)

print("success")