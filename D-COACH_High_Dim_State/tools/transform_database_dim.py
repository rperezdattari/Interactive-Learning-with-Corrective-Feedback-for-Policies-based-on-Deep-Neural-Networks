import numpy as np
import cv2
import time

database = np.load('../duckie_racing_database_120x160.npy')
database = np.reshape(database, [8000, 120, 160, 3])
print('database shape:', database.shape)
time.sleep(2)
new_database = []
new_size = 64

for i in range(database.shape[0]):
    image = cv2.resize(database[i], (new_size, new_size))
    new_database.append(image)
    print(np.shape(new_database))

np.save('../duckie_racing_database_64x64', new_database)
