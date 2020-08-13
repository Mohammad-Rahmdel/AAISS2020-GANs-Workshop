import tensorflow as tf


import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as animation
from sklearn.datasets import make_blobs
import argparse



def normalize(data):
    # X = X.numpy()
    X = np.zeros_like(data)
    X[:,0] = (data[:,0] - x_min) / (x_max - x_min)
    X[:,1] = (data[:,1] - y_min) / (y_max - y_min)
    X = X.astype("float32")
    return X


def get_networks(N_Z, DIMS):

    gen = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1 * 1 * 64, activation="relu", input_shape=(N_Z,)),
        tf.keras.layers.Dense(units=4 * 4, activation="relu"),
        tf.keras.layers.Dense(units=2, activation="sigmoid")
    ])

    
    disc = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=DIMS),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=100, activation='relu'),
        tf.keras.layers.Dense(units=10, activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])

    return gen, disc


# denormalizing - undo preprocessing
def denormalize(X):
    X = X.numpy()
    X[:,0] = X[:,0] * (x_max - x_min) + x_min
    X[:,1] = X[:,1] * (y_max - y_min) + y_min
    return X

def creat_circles(data, cntrs):
    dist = np.zeros(shape=(3,2))
    for i in range(data.shape[0]):
        d1 = np.linalg.norm(data[i] - cntrs[0])
        d2 = np.linalg.norm(data[i] - cntrs[1])
        d3 = np.linalg.norm(data[i] - cntrs[2])
        
        label = np.argmin([d1, d2, d3])
        distance = np.min([d1, d2, d3])
        dist[label, 0] += distance
        dist[label, 1] += 1
    
    radius = np.zeros(shape=(3))
    for i in range(3):
        radius[i] = dist[i, 0] / dist[i, 1]
        
    radius += 0.5

    return radius


def GAN_loss(y_true, y_pred):
    if adversarial_loss=='gan':
        return tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=True)
    elif adversarial_loss=='lsgan':
        return tf.keras.losses.mean_squared_error(y_true, y_pred)


@tf.function
def train_step(x):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        
        z_samp = tf.random.normal([x.shape[0], N_Z])

        x_gen = gen(z_samp)

        D_fake = disc(x_gen)
        D_real = disc(x)

        D_loss = GAN_loss(tf.ones_like(D_real), D_real) + GAN_loss(tf.zeros_like(D_fake), D_fake)
        G_loss = GAN_loss(tf.ones_like(D_fake), D_fake)
        
    G_gradients = gen_tape.gradient(G_loss, gen.trainable_variables)
    D_gradients = disc_tape.gradient(D_loss, disc.trainable_variables)
    
    gen_optimizer.apply_gradients(
    zip(G_gradients, gen.trainable_variables)
    )
    disc_optimizer.apply_gradients(
        zip(D_gradients, disc.trainable_variables)
    )
    
    return D_loss, G_loss

def train(n_epochs=1, print_=False):

    for epoch in range(n_epochs):
        Gloss = [] 
        Dloss = []
        for batch in train_dataset:
            D_loss, G_loss = train_step(batch)
            Gloss.append(G_loss)
            Dloss.append(D_loss)

        GLosses.append(np.mean(G_loss))
        DLosses.append(np.mean(D_loss))

        if print_:
            print(
            "Epoch: {} | disc_loss: {} | gen_loss: {}".format(
                epoch, DLosses[-1], GLosses[-1]
            ))


def draw(i, plot_interval, epoch, d, f, ax):
    ### train for plot_interval times
    train(plot_interval, False)
    d_pred = disc(z)
    d_pred = tf.reshape(d_pred, xx.shape)
    ### redraw discriminator prob
    d.set_array(d_pred)

    ### generate new data and rescatter
    z_samp = tf.random.normal([n, N_Z])
    x_gen = gen(z_samp)
    x_gen = denormalize(x_gen)
    f.set_offsets(np.c_[x_gen[:, 0], x_gen[:, 1]])
    
    ### update epoch number
    ax.set_title('epoch: ' + str(epoch + i * plot_interval + plot_interval))



def train_and_plot(k, plot_interval=1, epoch=0, frames=1000, interval=200):
    
    pause = True
    def onClick(event): 
        nonlocal pause
        if pause==True:
            pause = False
            anime.event_source.stop()
        else:
            pause = True
            anime.event_source.start()
        
    circle1 = plt.Circle(cntrs[0], radius[0], color='#339966', lw=2, fill=False)
    circle2 = plt.Circle(cntrs[1], radius[1], color='#339966', lw=2, fill=False)
    circle3 = plt.Circle(cntrs[2], radius[2], color='#339966', lw=2, fill=False)

    ### real data
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)

    
    plt.gcf().gca().add_artist(circle1)
    plt.gcf().gca().add_artist(circle2)
    plt.gcf().gca().add_artist(circle3)

    real = ax.scatter(data[:, 0], data[:, 1], 4, alpha=0.5, c='#800000',  marker='.') ##  #e60000
    ax.plot(cntrs[0][0], cntrs[0][1], 'o', markeredgecolor='k', markersize=5)
    ax.plot(cntrs[1][0], cntrs[1][1], 'o', markeredgecolor='k', markersize=5)
    ax.plot(cntrs[2][0], cntrs[2][1], 'o', markeredgecolor='k', markersize=5)


    ### discriminator
    # k = 2.5
    x_min, x_max = xmin - k, xmax + k
    y_min, y_max = ymin - k, ymax + k
    h = 0.1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    xx = (xx - x_min) / (x_max - x_min)
    yy = (yy - y_min) / (y_max - y_min)

    z = np.c_[xx.ravel(), yy.ravel()]
    z = z.astype('float32')
    
    ### draw discriminator output
    d_pred = disc(z)
    d_pred = tf.reshape(d_pred, xx.shape)
    d = plt.imshow(d_pred, vmin = 0., vmax = 1., cmap=plt.cm.coolwarm, origin='lower', alpha=0.7,
               extent=[x_min, x_max, y_min, y_max])


    ### generator - fake data
    z_samp = tf.random.normal([n, N_Z])
    x_gen = gen(z_samp)
    x_gen = denormalize(x_gen)

    # f = ax.scatter(x_gen[:, 0], x_gen[:, 1], 4, alpha=0.5, c='#1f1f7a', marker='.')
    f = ax.scatter(x_gen[:, 0], x_gen[:, 1], 4, alpha=0.5, c='#000000', marker='.')

    plt.legend([real, f], ['real', 'fake'], markerscale=7, loc='upper left',fontsize=12)

    plt.ylabel('Y')
    plt.xlabel('X')
    
    cbar = plt.colorbar(orientation='vertical')
    cbar.ax.set_yticklabels(['0 (fake)','0.2','0.4','0.6','0.8','1(real)'], fontsize=13)
    cbar.set_label('Discriminator Output', rotation=270, fontsize=15)
    
    
    ### play/pause with mouse click
    fig.canvas.mpl_connect('button_press_event', onClick) 

    anime = animation.FuncAnimation(fig, draw, fargs=(plot_interval, epoch, d, f, ax), frames=frames, interval=interval, repeat=False) 
    
    return anime




if __name__ == '__main__':

    tf.random.set_seed(5)   

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--adversarial_loss', type=str, default='gan', choices=['gan','lsgan'])
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--frames', type=int, default=60)
    parser.add_argument('--k', type=float, default=2.5)

    
    config = parser.parse_args()

    batch_size = config.batch_size
    adversarial_loss = config.adversarial_loss
    steps = config.steps
    frames = config.frames
    k = config.k

    N_Z = 64
    DIMS = (2)
    n = 2000
    cntrs = [[-2,-3],[10,0],[5,8]]
    data, _ = make_blobs(n_samples=n, centers=cntrs, n_features=2, random_state=3)

    xmin = min(data[:,0])
    xmax = max(data[:,0])
    ymin = min(data[:,1])
    ymax = max(data[:,1])

    # k = 2.5
    x_min, x_max = xmin - k, xmax + k
    y_min, y_max = ymin - k, ymax + k
    h = 0.1

    ### normalizing to 0-1
    X = normalize(data)

    gen, disc = get_networks(N_Z, DIMS)

    radius = creat_circles(data, cntrs)

    gen_optimizer = tf.keras.optimizers.Adam(0.001, beta_1=0.5)
    disc_optimizer = tf.keras.optimizers.Adam(0.001, beta_1=0.5)

    train_dataset = (
        tf.data.Dataset.from_tensor_slices(X)
        .shuffle(n)
        .batch(batch_size)
    )

    GLosses = []
    DLosses = []

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    z = np.c_[xx.ravel(), yy.ravel()]
    z = z.astype('float32')

    anim = train_and_plot(k, plot_interval=steps, epoch=0, frames=frames, interval=500)
    plt.show()
