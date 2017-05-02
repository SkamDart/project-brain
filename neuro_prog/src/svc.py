from sklearn.svm import SVC
import numpy as np

svc = SVC(kernel='linear')


class1_data = np.load('data/2BK-0BK_30-36-30.npz')['voxeldata']
#(30, 36, 30, 494)
class1_data = np.swapaxes(class1_data,0,3)
class1_data = np.reshape(class1_data, (494,-1))

class2_data = np.load('data/REL-MATCH_30-36-30.npz')['voxeldata']
#(30, 36, 30, 481)
class2_data = np.swapaxes(class2_data,0,3)
class2_data = np.reshape(class2_data, (481,-1))

training_data = np.concatenate((class1_data[:-10,:], class2_data[:-10,:]), axis=0)
testing_data = np.concatenate((class1_data[-10:,:], class2_data[-10:,:]), axis=0)

#print class1_data.shape, class2_data.shape, training_data.shape, testing_data.shape
training_label = np.concatenate((np.zeros(484),np.ones(471)))
testing_label = np.concatenate((np.zeros(10),np.ones(10)))

svc.fit(training_data, training_label)

pred = svc.predict(testing_data)
print pred
print testing_label
