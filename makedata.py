import numpy as np
from numpy.random import randint, choice, randn
import torch
import skimage.morphology
from scipy.ndimage import rotate
from skimage.morphology import ball, octahedron, cube
import elasticdeform as deform

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))

def make_data(num, datasize = (128, 128, 128)):
    data = np.zeros(datasize)
    r_l  = [randint(20, 21)                        for _ in range(num)]
    s_l  = [choice(['ball', 'octahedron', 'cube']) for _ in range(num)]
    z_l  = [randint(0, datasize[0])                for _ in range(num)]
    x_l  = [randint(0, datasize[1])                for _ in range(num)]
    y_l  = [randint(0, datasize[2])                for _ in range(num)]
    d_l  = [randn(3, 5, 5, 5)                      for _ in range(num)]
    for r, s, z, x, y, d in zip(r_l, s_l, z_l, x_l, y_l, d_l):
        form  = getattr(skimage.morphology, s)(r).astype(np.float32)
        form  = deform.deform_grid(X            = form ,
                                   displacement = d   ,)
        form  = form > 0.5
        z_max = min(z + form.shape[0], datasize[0])
        x_max = min(x + form.shape[1], datasize[1])
        y_max = min(y + form.shape[2], datasize[2])
        data[z : z + form.shape[0],
             x : x + form.shape[1],
             y : y + form.shape[2],]\
        += \
        form[0 : z_max - z        ,
             0 : x_max - x        ,
             0 : y_max - y        ,]
    data = data > 0
    data = torch.from_numpy(data)
    data = data.unsqueeze(0)
    return data

def make_beads_data(num, datasize = (128, 128, 128)):
    data = np.zeros(datasize)
    r_l  = [randint(5, 15)                         for _ in range(num)]
    s_l  = [choice(['ball', 'octahedron', 'cube']) for _ in range(num)]
    z_l  = [randint(0, datasize[0])                for _ in range(num)]
    x_l  = [randint(0, datasize[1])                for _ in range(num)]
    y_l  = [randint(0, datasize[2])                for _ in range(num)]
    d_l  = [randn(3, 5, 5, 5)                      for _ in range(num)]
    for r, s, z, x, y, d in zip(r_l, s_l, z_l, x_l, y_l, d_l):
        form  = getattr(skimage.morphology, s)(r).astype(np.float32)
        form  = deform.deform_grid(X=form, displacement=d*3,)
        form  = form > 0.5
        z_max = min(z + form.shape[0], datasize[0])
        x_max = min(x + form.shape[1], datasize[1])
        y_max = min(y + form.shape[2], datasize[2])
        data[z : z + form.shape[0],
             x : x + form.shape[1],
             y : y + form.shape[2],]\
        += \
        form[0 : z_max - z        ,
             0 : x_max - x        ,
             0 : y_max - y        ,]
    data = data > 0
    data = torch.from_numpy(data)
    data = data.unsqueeze(0)
    return data

def draw_3d_line(length):
    z = length
    x = 9
    y = 9
    arr = np.zeros((z, x, y))
    arr[:, x//2-1:x//2+2, y//2-1:y//2+2] = 1
    d = randn(3, 5, 1, 1)
    arr = deform.deform_grid(X=arr, displacement=d*3)
    r_arr = np.zeros((z, z, z))
    r_arr[:, (z-x)//2:(z-x)//2+x, (z-y)//2:(z-y)//2+y] = arr

    xd, yd, zd = np.random.random(3) * 360
    r_arr0 = rotate(r_arr , yd, axes=(1, 0), reshape=False, order=1, mode='nearest', cval=0.0)
    r_arr1 = rotate(r_arr0, xd, axes=(2, 0), reshape=False, order=1, mode='nearest', cval=0.0)
    r_arr2 = rotate(r_arr1, zd, axes=(2, 1), reshape=False, order=1, mode='nearest', cval=0.0)
    r_arr3 = r_arr2 > 0.1
    return r_arr3

def draw_thick_3d_line(length):
    z = length
    x = 11
    y = 11
    arr = np.zeros((z, x, y))
    arr[:, x//2-3:x//2+4, y//2-3:y//2+4] = 1
    d = randn(3, 5, 1, 1)
    arr = deform.deform_grid(X=arr, displacement=d*3)
    r_arr = np.zeros((z, z, z))
    r_arr[:, (z-x)//2:(z-x)//2+x, (z-y)//2:(z-y)//2+y] = arr

    xd, yd, zd = np.random.random(3) * 360
    r_arr0 = rotate(r_arr , yd, axes=(1, 0), reshape=False, order=1, mode='nearest', cval=0.0)
    r_arr1 = rotate(r_arr0, xd, axes=(2, 0), reshape=False, order=1, mode='nearest', cval=0.0)
    r_arr2 = rotate(r_arr1, zd, axes=(2, 1), reshape=False, order=1, mode='nearest', cval=0.0)
    r_arr3 = r_arr2 > 0.1
    return r_arr3

def draw_thick_3d_line_fixed_z_angle(length, zangle, deterministic=True):
    z = length
    x = 11
    y = 11
    arr = np.zeros((z, x, y))
    arr[:, x//2-3:x//2+4, y//2-3:y//2+4] = 1
    d = randn(3, 5, 1, 1)
    arr = deform.deform_grid(X=arr, displacement=d*3)
    r_arr = np.zeros((z, z, z))
    r_arr[:, (z-x)//2:(z-x)//2+x, (z-y)//2:(z-y)//2+y] = arr

    xd, yd = np.random.random(2) * 360
    r_arr0 = rotate(r_arr , yd,     axes=(1, 0), reshape=False, order=1, mode='nearest', cval=0.0)
    r_arr1 = rotate(r_arr0, xd,     axes=(2, 0), reshape=False, order=1, mode='nearest', cval=0.0)
    r_arr2 = rotate(r_arr1, zangle, axes=(2, 1), reshape=False, order=1, mode='nearest', cval=0.0)
    r_arr3 = r_arr2 > 0.1
    return r_arr3

def make_realistic_data(num, datasize = (128, 128, 128), mu=0, sigma=0.5):
    data_x = np.zeros(datasize)
    data_z = np.zeros(datasize)
    r_l  = [randint(10, 30)                        for _ in range(num)]
    lu_l = [mu + sigma * randn(1)                  for _ in range(num)]
    l_l  = [randint(20, 120)                       for _ in range(num)]
    s_l  = [choice(['ball', 'octahedron', 'cube', 'line'],
                   p=[1/30, 1/30, 1/30, 9/10])     for _ in range(num)]
    z_l  = [randint(0, datasize[0])                for _ in range(num)]
    x_l  = [randint(0, datasize[1])                for _ in range(num)]
    y_l  = [randint(0, datasize[2])                for _ in range(num)]
    d_l  = [randn(3, 5, 5, 5)                      for _ in range(num)]
    for r, l, lu, s, z, x, y, d in zip(r_l, l_l, lu_l, s_l, z_l, x_l, y_l, d_l):
        if s != 'line':
            form  = getattr(skimage.morphology, s)(r).astype(np.float32)
            form  = deform.deform_grid(X=form, displacement=d*3,)
            form  = form > 0.5
        else:
            form = draw_3d_line(l)
        z_max = min(z + form.shape[0], datasize[0])
        x_max = min(x + form.shape[1], datasize[1])
        y_max = min(y + form.shape[2], datasize[2])

        data_x[z : z + form.shape[0],
               x : x + form.shape[1],
               y : y + form.shape[2],]\
        += \
        form[0 : z_max - z        ,
             0 : x_max - x        ,
             0 : y_max - y        ,]

        data_z[z : z + form.shape[0],
             x : x + form.shape[1],
             y : y + form.shape[2],]\
        = np.maximum(
            data_z[z : z + form.shape[0],
                   x : x + form.shape[1],
                   y : y + form.shape[2],],
            form  [0 : z_max - z,
                   0 : x_max - x,
                   0 : y_max - y,] * min(1., np.exp(lu)))
    data_x = data_x > 0.5
    data_x = data_x.astype(np.float32)[None]
    data_z = data_z.astype(np.float32)[None]
    return {"data_x": data_x,
            "data_z": data_z}

def make_thick_realistic_data(num, datasize = (128, 128, 128),
                              mu=0, sigma=0.5, p=0.1):
    data_x = np.zeros(datasize)
    data_z = np.zeros(datasize)
    r_l  = [randint(10, 30)                        for _ in range(num)]
    lu_l = [mu + sigma * randn(1)                  for _ in range(num)]
    l_l  = [randint(120, 240)                       for _ in range(num)]
    s_l  = [choice(['ball', 'octahedron', 'cube', 'line'],
                   p=[p/3, p/3, p/3, 1-p])     for _ in range(num)]
    z_l  = [randint(0, datasize[0])                for _ in range(num)]
    x_l  = [randint(0, datasize[1])                for _ in range(num)]
    y_l  = [randint(0, datasize[2])                for _ in range(num)]
    d_l  = [randn(3, 5, 5, 5)                      for _ in range(num)]
    for r, l, lu, s, z, x, y, d in zip(r_l, l_l, lu_l, s_l, z_l, x_l, y_l, d_l):
        if s != 'line':
            form  = getattr(skimage.morphology, s)(r).astype(np.float32)
            form  = deform.deform_grid(X=form, displacement=d*3,)
            form  = form > 0.5
        else:
            form = draw_thick_3d_line(l)
        z_max = min(z + form.shape[0], datasize[0])
        x_max = min(x + form.shape[1], datasize[1])
        y_max = min(y + form.shape[2], datasize[2])

        data_x[z : z + form.shape[0],
               x : x + form.shape[1],
               y : y + form.shape[2],]\
        += \
        form[0 : z_max - z        ,
             0 : x_max - x        ,
             0 : y_max - y        ,]

        data_z[z : z + form.shape[0],
             x : x + form.shape[1],
             y : y + form.shape[2],]\
        = np.maximum(
            data_z[z : z + form.shape[0],
                   x : x + form.shape[1],
                   y : y + form.shape[2],],
            form  [0 : z_max - z,
                   0 : x_max - x,
                   0 : y_max - y,] * min(1., np.exp(lu)))
    data_x = data_x > 0.5
    data_x = data_x.astype(np.float32)[None]
    data_z = data_z.astype(np.float32)[None]
    return {"data_x": data_x,
            "data_z": data_z}

def draw_thick_3d_line_fixed_z_angle(length, zangle):
    z = length
    x = 11
    y = 11
    arr = np.zeros((z, x, y))
    arr[:, x//2-3:x//2+4, y//2-3:y//2+4] = 1
    d = randn(3, 5, 1, 1)
    arr = deform.deform_grid(X=arr, displacement=d*3)
    r_arr = np.zeros((z, z, z))
    r_arr[:, (z-x)//2:(z-x)//2+x, (z-y)//2:(z-y)//2+y] = arr

    xd,zd= np.random.uniform(low=0, high=360, size=2)
    if zangle == 'random':
        zangle = zd
    r_arr0 = rotate(r_arr , zangle-90, axes=(2, 0),
                    reshape=False, order=1, mode='nearest', cval=0.0)
    r_arr1 = rotate(r_arr0, xd, axes=(1, 2),
                    reshape=False, order=1, mode='nearest', cval=0.0) 
    r_arr2 = r_arr1 > 0.1
    return r_arr2

def make_determined_zangle_data(num, datasize = (128, 128, 128), angle=45):
    data_x = np.zeros(datasize)
    l_l  = [randint(20, 40)                       for _ in range(num)]
    z_l  = [randint(0, datasize[0])                for _ in range(num)]
    x_l  = [randint(0, datasize[1])                for _ in range(num)]
    y_l  = [randint(0, datasize[2])                for _ in range(num)]
    for l, z, x, y in zip(l_l, z_l, x_l, y_l):

        form = draw_thick_3d_line_fixed_z_angle(l, angle)
        z_max = min(z + form.shape[0], datasize[0])
        x_max = min(x + form.shape[1], datasize[1])
        y_max = min(y + form.shape[2], datasize[2])

        data_x[z : z + form.shape[0],
               x : x + form.shape[1],
               y : y + form.shape[2],]\
        += \
        form[0 : z_max - z        ,
             0 : x_max - x        ,
             0 : y_max - y        ,]

    data_x = data_x > 0.5
    data_x = (data_x*(2*16-1)).astype(np.uint16)
    return {"data_x": data_x}