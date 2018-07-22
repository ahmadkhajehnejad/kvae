import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import os

plt.rcParams['animation.ffmpeg_path'] = u'/usr/bin/ffmpeg'

def save_frames_from_pickle(filename):
    
    with open(filename + '._.pkl', 'rb') as fin:
        images = pickle.load(fin)
    
    num_sequences, n_steps, w, h = images.shape

    fig = plt.figure()
    im = plt.imshow(combine_multiple_img(images[:, 0]), cmap=plt.cm.get_cmap('Greys'), interpolation='none')
    plt.axis('image')

    def updatefig(*args):
        im.set_array(combine_multiple_img(images[:, args[0]]))
        return im,

    ani = animation.FuncAnimation(fig, updatefig, interval=500, frames=n_steps)

    # Either avconv or ffmpeg need to be installed in the system to produce the videos!
    try:
        writer = animation.writers['avconv']
    except KeyError:
        writer = animation.writers['ffmpeg']
    writer = writer(fps=3)
    ani.save(filename, writer=writer)
    plt.close(fig)


def save_true_generated_frames_from_pickle(filename):
    with open(filename + '._.pkl', 'rb') as fin:
        true, generated = pickle.load(fin)
    
    num_sequences, n_steps, w, h = true.shape

    # Background is 0, foreground as 1
    true = np.copy(true[:16])
    true[true > 0.1] = 1

    # Set foreground be near 0.5
    generated = generated * .5

    # Background is 1, foreground is near 0.5
    generated = 1 - generated[:16, :n_steps]

    # Subtract true from generated so background is 1, true foreground is 0,
    # and generated foreground is around 0.5
    images = generated - true
    # images[images > 0.5] = 1.

    fig = plt.figure()
    im = plt.imshow(combine_multiple_img(images[:, 0]), cmap=plt.cm.get_cmap('gist_heat'),
                    interpolation='none', vmin=0, vmax=1)
    plt.axis('image')

    def updatefig(*args):
        im.set_array(combine_multiple_img(images[:, args[0]]))
        return im,

    ani = animation.FuncAnimation(fig, updatefig, interval=500, frames=n_steps)

    try:
        writer = animation.writers['avconv']
    except KeyError:
        writer = animation.writers['ffmpeg']
    writer = writer(fps=3)
    ani.save(filename, writer=writer)
    plt.close(fig)


def combine_multiple_img(images, table_size=4, indices=None):

    if indices is None:
        indices = range(table_size**2)

    i = 0
    height = images[0].shape[0]
    width = images[0].shape[1]
    img_out = np.zeros((height * table_size, width * table_size))
    for x in range(table_size):
        for y in range(table_size):
            xa, xb = x * height, (x + 1) * height
            ya, yb = y * width, (y + 1) * width
            img_out[xa:xb, ya:yb] = images[indices[i]]
            i += 1

    return img_out


base_dir = 'results/orig_box_gravity/'
all_files = os.listdir(base_dir)
pkl_files = [f[:-6] for f in all_files if f[-6:] == '._.pkl']

for f in pkl_files:
    if f[:11] == 'video_true_':
        save_true_generated_frames_from_pickle(base_dir + f)
    else:
        save_frames_from_pickle(base_dir + f)
