import numpy as np
from PIL import Image
import os
import re
from sklearn.preprocessing import LabelEncoder

#only match the first two regex groups in the file name. Goes from 1 to 4
greebles_mode = 2

path = r"D:\Users\Enzo\Downloads\greebles-generator-master\images"
path_train = path + "\\train"
path_test = path + "\\test"

train_filenames = os.listdir(path_train)
test_filenames = os.listdir(path_test)

# t_size = (len(train_filenames), 32, 32, 3)
# train_images = np.empty(t_size)
# for idx,fname in enumerate(train_filenames):
    # image = Image.open(path_train + "\\" + fname)
    # image = image.convert("RGB")
    # image = np.asarray(image, dtype=np.float32) / 255
    # train_images[idx] = image[:, :, :3]

# Remove alpha channel from png file, just keep the first 3 channels
train_images = np.array([np.array(Image.open(path_train + "\\" + fname))[...,:3] for fname in train_filenames])
test_images = np.array([np.array(Image.open(path_test + "\\" + fname))[...,:3] for fname in test_filenames])

'''
File names denote the individual Greeble by defining the specific origin of the body type and parts, as well as its gender.

The first character is the gender (m/f)

The second number is the family (defined by body type, 1-5)
Next there is a tilda (~) (is this referring to the dash in the filename?)

The next few numbers describe where the parts came from in terms of the original Greebles.

The third number is the family these particular parts ORIGINALLY came from. That is, a "2" would denote that the parts in the Greeble you are dealing with came from family 2 (1-5)

The final number is which set of parts were taken from the specified family. Note that genders are never crossed (!), so that the number here only refers to the same gender parts as the Greeble you are dealing with. Depending on the number of individual Greebles in the original set, there could more more or less of these part sets (1-10, where 10 is the max possible as of August 2002).

For example, "f1~16.max" is the model of a female Greeble of family 1, with body parts from family 1, set 6.
'''

train_labels = np.zeros(len(train_filenames), dtype='int32')
test_labels = np.zeros(len(test_filenames), dtype='int32')

train_labels_temp = np.zeros(len(train_filenames), dtype=object)
for idx, fname in enumerate(train_filenames):
    l = np.empty(greebles_mode,dtype=object)
    s = "-"
    #replace all non alphanumeric characters with nothing
    label = re.sub('[^A-Za-z0-9]+', '', fname)
    #match label structure
    matchObj = re.match( r'(f|m)([1-5]{1})([1-5]{1})(10|[1-9])', label, re.M|re.I)
    if matchObj:
        #male of female
        if(greebles_mode >=1):
            l[0] = matchObj.group(1)
        #body type, 1-5
        if(greebles_mode >=2):
            l[1] = matchObj.group(2)
        #original family, 1-5
        if(greebles_mode >=3):
            l[2] = matchObj.group(3)
        #which set of parts, 1-10
        if(greebles_mode >=4):
            l[3] = matchObj.group(4)
    else:
       raise NameError('Wrong file name structurem check the greebles documentation.')
    s = s.join(l)
    train_labels_temp[idx] = s

test_labels_temp = np.zeros(len(test_filenames), dtype=object)
for idx, fname in enumerate(test_filenames):
    l = np.empty(greebles_mode,dtype=object)
    s = "-"
    #replace all non alphanumeric characters with nothing
    label = re.sub('[^A-Za-z0-9]+', '', fname)
    #match label structure
    matchObj = re.match( r'(f|m)([1-5]{1})([1-5]{1})(10|[1-9])', label, re.M|re.I)
    if matchObj:
        #male of female
        if(greebles_mode >=1):
            l[0] = matchObj.group(1)
        #body type, 1-5
        if(greebles_mode >=2):
            l[1] = matchObj.group(2)
        #original family, 1-5
        if(greebles_mode >=3):
            l[2] = matchObj.group(3)
        #which set of parts, 1-10
        if(greebles_mode >=4):
            l[3] = matchObj.group(4)
    else:
       raise NameError('Wrong file name structurem check the greebles documentation.')
    s = s.join(l)
    test_labels_temp[idx] = s

le = LabelEncoder()
train_labels = np.asarray(le.fit_transform(train_labels_temp))
test_labels = np.asarray(le.fit_transform(test_labels_temp))


# Test image
# train_labels_temp[7400]
# train_filenames[7400]
# new_im = Image.fromarray(train_images[7400])
# new_im.save("numpy_altered_sample2.png")