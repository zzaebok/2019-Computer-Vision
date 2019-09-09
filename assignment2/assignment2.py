import numpy as np
import cv2
import os
import sys


STUDENT_CODE = '2014121065'
FILE_NAME = 'output.txt'
if not os.path.exists(STUDENT_CODE):
    os.mkdir(STUDENT_CODE)
f = open(os.path.join(STUDENT_CODE, FILE_NAME), 'w')
criteria = float(sys.argv[1])
dir_path = 'faces_training'
test_dir_path = 'faces_test'
test_imgs = os.listdir(test_dir_path)
train_imgs = os.listdir(dir_path)
x = np.empty([39, 8064])
for i, train_img in enumerate(train_imgs):
    img = cv2.imread(os.path.join(dir_path, train_img), cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32)
    img = img.flatten()
    x[i] = img

mean = np.mean(x, axis=0)
x -= mean
x = x.T
u, s, _ = np.linalg.svd(x)

#compute eigen values and number of principal compnents k
ss = np.square(s)
ss = ss/ss.sum()
k = 0
while criteria > 0:
    criteria -= ss[k]
    k += 1

f.write('########## STEP 1 ##########\n')
f.write('Input Percentage: '+ sys.argv[1] +'\n')
f.write('Selected Dimension: '+str(k)+'\n\n')

w = u[:,:k]
#coefficients
y = np.matmul(np.transpose(w), x)
#projected vectors
res = np.matmul(w, y)

tmp_res = res.copy()
#restore
#vertical to horizontal
x = x.T
x += mean
#vertical
x = x.T

#vertical to horizontal
res = res.T
res += mean
#vertical
res = res.T

#errors = horizontal
errors = np.mean(np.square(res-x), axis=0)
average_error = np.mean(errors)

#res = res.astype(np.uint8)
res = res.T
for i, re in enumerate(res):
    cv2.imwrite(os.path.join(STUDENT_CODE, train_imgs[i]), re.reshape((96,84)))

f.write('########## STEP 2 ##########\n')
f.write('Reconstruction error\n')
f.write('average : '+ str(round(average_error,4)) + '\n')
for i in range(39):
    if i<9:
        f.write('0')
    f.write(str(i+1) + ': ' + str(round(errors[i],4)) +  '\n')
f.write('\n')

test_x = np.empty([5, 8064])
for i, test_img in enumerate(test_imgs):
    img = cv2.imread(os.path.join(test_dir_path, test_img), cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32)
    img = img.flatten()
    test_x[i] = img

#test_mean = np.mean(test_x, axis=0)
test_x -= mean
test_x = test_x.T

test_y = np.matmul(np.transpose(w), test_x)
test_res = np.matmul(w, test_y)
test_res = test_res.T
f.write('########## STEP 3 ##########\n')
for i in range(5):
    l2_distance = np.sqrt(np.sum(np.square(tmp_res.T - test_res[i]), axis=1))
    f.write('{} ==> {}\n'.format(test_imgs[i], train_imgs[np.argmin(l2_distance)]))

f.close()