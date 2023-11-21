#!/usr/bin/env python
# coding: utf-8

# # Day 8

# # 2. simple-latent-space

# In[2]:


import fastai
fastai.__version__


# In[3]:


from fastai.test_utils import show_install
show_install()


# In[4]:


from fastai.vision.all import *
import numpy as np
import math


# In[5]:


PATTERN_SIZE=20
patterns_labels = ["cos", "sin"]
# patterns = [ np.zeros(PATTERN_SIZE), np.ones(PATTERN_SIZE),  np.array([math.sin(i*math.pi/PATTERN_SIZE)for i in range(0, PATTERN_SIZE)])]
patterns = [ [math.cos(i*math.pi/PATTERN_SIZE)for i in range(0, PATTERN_SIZE)], [math.sin(i*math.pi/PATTERN_SIZE)for i in range(0, PATTERN_SIZE)]]
patterns


# In[6]:


items = []
for i in progress_bar(range(0, 10000)):
    items += [((patterns[i % len(patterns)] +  np.random.normal(0, .1, PATTERN_SIZE)).astype(float), patterns_labels[(i % len(patterns))],i % len(patterns))]


# In[7]:


# Tensor(items[0][0].tolist()).float()
items[2][2]


# In[8]:


# items = list(range(0,10))
def get_items(_):
    return list(range(0, len(items)))
#     return 

def label_func(i):
    return items[i][1]
def get_y(i):
    pos = items[i][2]
    res = [0]*len(patterns)
    res[pos]=1
    return Tensor(res).float()
def get_x(i):
    item = items[i]
    a = item[0].tolist()
    return Tensor(a).float()


# In[9]:


dblock = DataBlock(blocks= (TransformBlock, CategoryBlock), get_items=get_items, get_y=label_func, get_x=get_x, splitter  = RandomSplitter())
# dblock = DataBlock(blocks= (TransformBlock, TransformBlock), get_items=get_items, get_y=get_y, get_x=get_x, splitter  = RandomSplitter())
dsets = dblock.datasets(10)
dsets.train[0]


# In[10]:


dls = dblock.dataloaders(0)
x,y = dls.one_batch()
x.shape, y.shape


# In[11]:


def encoder():
    return nn.Sequential(
        nn.Linear(PATTERN_SIZE,8),
        nn.ReLU(),
        nn.Linear(8,2)
    )

def classifier():
    return nn.Sequential(
        encoder(), 
        nn.Sequential(
            nn.ReLU(),
            nn.Linear(2,1),
            nn.Sigmoid(),
        )
    )


# In[12]:


cla = classifier()
cla(x).shape


# In[13]:


learn = Learner(dls, cla, loss_func=MSELossFlat())


# In[14]:


learn.lr_find()


# In[15]:


learn.fine_tune(5, base_lr = 0.04)


# In[16]:


with torch.no_grad():
    for i in range(0,6):
        print(items[i])
        print(cla(get_x(i)))


# In[17]:


dblock = DataBlock(blocks= (TransformBlock, TransformBlock), get_items=get_items, get_x=get_x, get_y=get_x, splitter  = RandomSplitter())
# dblock = DataBlock(blocks= (TransformBlock, CategoryBlock), get_items=get_items, get_y=label_func, get_x=get_x, splitter  = RandomSplitter())
dsets = dblock.datasets(10)
dsets.train[0]


# In[18]:


dls = dblock.dataloaders(0)
x,y = dls.one_batch()
x.shape, y.shape


# In[19]:


def encoder():
    return nn.Sequential(
        nn.Linear(PATTERN_SIZE,8),
        nn.ReLU(),
        nn.Linear(8,2)
    )
def decoder():
    return nn.Sequential(
        nn.Linear(2,8),
        nn.ReLU(),
        nn.Linear(8,PATTERN_SIZE)
    )

def autoencoder(): return nn.Sequential(encoder(), decoder())


# In[20]:


ac = autoencoder()
enc = encoder()
dec = decoder()


# In[21]:


ac(x).shape


# In[22]:


dec(enc(x)).shape


# In[23]:


learn = Learner(dls, ac, loss_func=MSELossFlat())


# In[24]:


learn.lr_find()


# In[25]:


learn.fit_one_cycle(25, 0.04)


# In[26]:


from sklearn.metrics import mean_squared_error


# In[27]:


i=0
with torch.no_grad():
    print(ac(get_x(i)))
    print(items[i][0])
    rms = mean_squared_error(items[i][0], ac(get_x(0)), squared=False)
    print(rms)


# In[28]:


with torch.no_grad():
    for i in range(0,6):
        print("*"*50)
        print(items[i])
        print("=>")
        print(ac(get_x(i)))
        rms = mean_squared_error(items[i][0], ac(get_x(i)), squared=False)
        print(f"rms: {rms}")


# In[29]:


import seaborn as sns
# tips = sns.load_dataset("tips")
# tips.head()


# In[30]:


with torch.no_grad():
    result = []
    for i in progress_bar(range(0, len(items))):
        item = items[i]
        result += [ (item[1], *enc(get_x(i)).numpy())]


# In[31]:


df = pd.DataFrame(result, columns=["c","x", "y"])
df


# In[32]:


sns.scatterplot(data=df, x="x", y="y", hue="c")


# # 1.

# In[33]:


# CIFAR10 DCGAN Example
# based on https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/

from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.datasets.cifar10 import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from matplotlib import pyplot


# define the standalone discriminator model
def define_discriminator(in_shape=(32, 32, 3)):
    model = Sequential()
    # normal
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # downsample
    model.add(Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # classifier
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# define the standalone generator model
def define_generator(latent_dim):
    model = Sequential()
    # foundation for 4x4 image
    n_nodes = 256 * 4 * 4
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4, 4, 256)))
    # upsample to 8x8
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 16x16
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 32x32
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # output layer
    model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))
    return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(g_model)
    # add the discriminator
    model.add(d_model)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


# load and prepare cifar10 training images
def load_real_samples():
    # load cifar10 dataset
    (trainX, _), (_, _) = load_data()
    # convert from unsigned ints to floats
    X = trainX.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    return X


# select real samples
def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, 1))
    return X, y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    X = g_model.predict(x_input)
    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return X, y


# create and save a plot of generated images
def save_plot(examples, epoch, n=7):
    # scale from [-1,1] to [0,1]
    examples = (examples + 1) / 2.0
    # plot images
    for i in range(n * n):
        # define subplot
        pyplot.subplot(n, n, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(examples[i])
    # save plot to file
    # filename = 'generated_plot_e%03d.png' % (epoch+1)
    # pyplot.savefig(filename)
    pyplot.show()


# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):
    # prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    # summarize discriminator performance
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real * 100, acc_fake * 100))
    # save plot
    save_plot(x_fake, epoch)
    # save the generator model tile file
    filename = 'generator_model.h5'
    g_model.save(filename)


# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=200, n_batch=128):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            if j % 100 == 0:
                print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' % (i + 1, j + 1, bat_per_epo, d_loss1, d_loss2, g_loss))
        # evaluate the model performance, sometimes
        # if (i+1) % 10 == 0:
        summarize_performance(i, g_model, d_model, dataset, latent_dim)


# size of the latent space
latent_dim = 100
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
dataset = load_real_samples()
# train model
train(g_model, d_model, gan_model, dataset, latent_dim, 20, 256)


# In[ ]:




