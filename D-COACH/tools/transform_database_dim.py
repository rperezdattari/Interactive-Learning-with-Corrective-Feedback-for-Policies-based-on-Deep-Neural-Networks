import numpy as np
import cv2
import time


new_size = 64  # Define new image height/width

database = np.load('racing_car_database.npy')
database_length = database.shape[0]
old_size = database.shape[2]
database = np.reshape(database, [database_length, old_size, old_size, 3])
print('database shape:', database.shape)
time.sleep(2)
new_database = []

for i in range(database_length):
    image = cv2.resize(database[i], (new_size, new_size))
    new_database.append(image)
    print(np.shape(new_database))

np.save('racing_car_database_%sx%s' % (new_size, new_size), new_database)
