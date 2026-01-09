import os
import cv2
import numpy as np

data = []
labels = []

base_dir = "."   # current directory
folders = ["Kartik", "Jasveer", "Manya", "jassi"]

i = 1

for label, folder in enumerate(folders):
    folder_path = os.path.join(base_dir, folder)

    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)

            if img is not None:
                img = cv2.resize(img, (224, 224))
                img = img / 255.0

                data.append(img)
                labels.append(label)
                print(i)
                i = i + 1

X = np.array(data)
y = np.array(labels)

np.save("X.npy", X)
np.save("y.npy", y)


print("Images shape:", X.shape)
print("Labels shape:", y.shape)
