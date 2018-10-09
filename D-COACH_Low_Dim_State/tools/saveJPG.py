import matplotlib.pyplot as plt
import numpy as np

database_size = 11708
database = np.array([])
for image_n in range(database_size):
    if image_n % 100 == 0 and image_n != 0:
        print(str(image_n)+' images imported...')
        np.save('racing_car_database', database)
        del database
        database = np.load('racing_car_database.npy')
        print(database.shape)
    try:
        image = plt.imread('/home/rodrigo/Documents/Tesis/Racing Car/racing_car_database/test'+str(image_n)+'.jpeg')
        print('size', image.reshape([96, 96, 3]))
        image = np.expand_dims(image, axis=0)
    except:
        print('image: ' + str(image_n) + ' not found.')
        continue
    if np.shape(database)[0] == 0:
        database = image
    else:
        database = np.append(database, image, axis=0)

print(database.shape)

np.save('racing_car_database', database)

#im = np.load('racing_car_database.npy')

#plt.imshow(im)
#plt.show()