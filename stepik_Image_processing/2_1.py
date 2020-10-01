#Запускает изображение в браузере, но так как я пользуюсь Atom, мне она не нужна.
%matplotlib inline

from skimage.io import imread, imshow, imsave

# %% codecell
tiger = imread(r'C:\programming\Data_Science\working\stepik_Image_processing\tiger-color.png')
imshow(tiger)

# %% codecell
#Получаем число строк, столбцов и количество каналов
tiger.shape
#обратимся к пикселю на носу
#uint8 - без знаковый тип число от 0 до 255
tiger[383, 374]
#получаем синюю компоненту
tiger[383, 374, 2]
#меняем цвет пикселя на жолтый - [255, 255, 0]
tiger_nose = tiger.copy()
tiger_nose[383, 374] = [255, 255, 0]
#Сохраняем изображение
imsave('tiger-yellow-nose.png', tiger_nose)

# %% codecell
img = imread(r'C:\programming\Data_Science\working\stepik_Image_processing\img.png')
print(img.shape[1])

# %% codecell
#tiger = imread('img.png')
x = (tiger.shape[0] // 2)
y = (tiger.shape[1] // 2)

tiger_nose = tiger.copy()
tiger_nose[x, y] = [102, 204, 102]

imsave('out_img1.png', tiger_nose)

# %% codecell
nose = tiger[370:410, 350:440]
imshow(nose)

tiger_nose = tiger.copy()
tiger_nose[370:410, 350:440] = [255, 0, 255]

imshow(tiger)

img_yellow = imread('tiger-yellow-nose.png')
imshow(img_yellow[370:410, 350:440])

# %% codecell
#tiger = imread('img.png')
x = (tiger.shape[0] // 2) - 3
y = (tiger.shape[1] // 2) - 7

tiger_nose = tiger.copy()
tiger_nose[x, y]

tiger_nose[x: (x + 7), y: (y + 15)] = [255, 192, 203]
imshow(tiger_nose)

imsave('out_img2.png', tiger_nose)

# %% codecell
tiger_border = imread(r'C:\programming\Data_Science\working\stepik_Image_processing\tiger-border.png')
#tiger_border = imread('img.png')

border_color = tiger_border[0, 0]

x = (tiger_border.shape[0] // 2)
y = (tiger_border.shape[1] // 2)

for xi_border in range(tiger_border.shape[0]):
    if all(tiger_border[xi_border, y] != border_color):
        x_top_border = xi_border
        break

for xi_border in reversed(range(tiger_border.shape[0])):
    if all(tiger_border[xi_border, y] != border_color):
        x_below_border = xi_border + 1
        break
x_below_border = tiger_border.shape[0] - x_below_border

for yi_border in range(tiger_border.shape[1]):
    if all(tiger_border[x, yi_border] != border_color):
        y_left_border = yi_border
        break

for yi_border in reversed(range(tiger_border.shape[1])):
    if all(tiger_border[x, yi_border] != border_color):
        y_below_border = yi_border + 1
        break
y_below_border = tiger_border.shape[1] - y_below_border

print(y_left_border ,x_top_border ,y_below_border ,x_below_border)

# %% codecell
from numpy import *
from skimage.io import *
a=where(tiger_border != tiger_border[0,0])
print("%i %i %i %i"%(a[1][0], a[0][0], tiger_border.shape[1]-1-a[1][-1], tiger_border.shape[0]-1-a[0][-1]))
