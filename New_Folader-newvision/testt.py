import numpy as np
label = np.load("/DATA/DATA2/lj/UrbanLF/semantic_segmentation/UrbanLF-Real/train/Image768/label.npy")
print(label.shape)
positions = np.where(label == 14)

# print(list(zip(positions[0], positions[1])))  # [(2, 1)]

# print(sum(label==14))
print(np.unique(label))