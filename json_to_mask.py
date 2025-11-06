import numpy as np
import cv2
import json

#########============  Converts json files saved from imglab after annotation tool to mask ===========###########

# Read the file - as simple txt file
f = open("/home/volcani/agribot_ws/Unet/train_5_11.json", "r")
data = json.load(f)  # Load the data

img_dir = "/home/volcani/agribot_ws/Unet/field_images/plowing/train_5_11"
mask_dir = "/home/volcani/agribot_ws/Unet/field_images/plowing/mask_train_5_11"

images = data["images"]
annots = data["annotations"]

# Create a dictionary to map image_id to image info
image_dict = {img["id"]: img for img in images}

print(f"Total images: {len(images)}")
print(f"Total annotations: {len(annots)}")

# Process each annotation and match it with the correct image
for annot in annots:
    image_id = annot["image_id"]
    
    # Get the corresponding image info
    if image_id not in image_dict:
        print(f"Warning: No image found for annotation with image_id {image_id}")
        continue
    
    img_info = image_dict[image_id]
    filename = img_info["file_name"]
    h = img_info["height"]
    w = img_info["width"]

    # Create empty mask
    mask = np.zeros((h, w))

    seg = annot["segmentation"]

    for points in seg:
        contours = []

        for i in range(0, len(points), 2):
            contours.append((points[i], points[i+1]))
        
        contours = np.array(contours, dtype=np.int32)

        cv2.drawContours(mask, [contours], -1, 255, -1)

    # Save with the correct filename
    cv2.imwrite(f"{mask_dir}/{filename}", mask)
    print(f"Saved mask for {filename} (image_id: {image_id})")































# import numpy as np
# import cv2
# import json

# #########============  Converts jason files saved from imglab after anotation tool to mask ===========###########


# # Read the file - as simple txt file
# f = open("/home/volcani/agribot_ws/Unet/train.json", "r")
# data = json.load(f) # Load the data

# # print(data)

# img_dir = "/home/volcani/field_images/plowing/train"
# mask_dir = "/home/volcani/field_images/plowing/mask_train"

# images = data["images"]
# annots = data["annotations"]

# print(data)

# for x, y in zip(images, annots):
#     filename = x["file_name"]
#     h = x["height"]
#     w = x["width"]

#     mask = np.zeros((h, w))

#     seg = y["segmentation"]

#     for points in seg:
#         contours =[]

#         for i in range(0, len(points), 2):
#             contours.append((points[i], points[i+1]))
        
#         contours = np.array(contours, dtype=np.int32)

#         cv2.drawContours(mask, [contours], -1, 255, -1)

#     cv2.imwrite(f"{mask_dir}/{filename}", mask)        