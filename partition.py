import os
import shutil
import random
import collections

#Getting the file names of all images
src = "training_images"
img_files = os.listdir(src)
print(img_files)
print(len(img_files))

print([item for item, count in collections.Counter(img_files).items() if count > 1])
#No duplicates

train_dst = "train_images"
test_dst = "test_images"
#Getting 90% random images as train set
for i in range(900):
  idx = random.randint(0, len(img_files)-1)
  filename = img_files[idx]
  src_name = os.path.join(src, filename)
  print(src_name)
  shutil.move(src_name, train_dst)
  img_files.remove(img_files[idx])

#Getting the remaining 10% as test set
for filename in img_files:
  src_name = os.path.join(src, filename)
  print(src_name)
  shutil.move(src_name, test_dst)
