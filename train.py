import tensorflow as tf
import os
import matplotlib.pyplot as plt
import time
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU
from keras.layers import Conv2DTranspose, Dropout, ReLU, Input, Concatenate, ZeroPadding2D
from keras.optimizers import Adam


devices = tf.config.experimental.list_physical_devices("GPU")

BATCH_SIZE = 1
IMAGE_SIZE = 256

#Make Folder before use
output_folder = r'C:\Users\rajat\Downloads\final_dataset (train+val)\Output'

pathHazy = r'C:\Users\rajat\Downloads\final_dataset (train+val)\train\hazy'
pathGT = r'C:\Users\rajat\Downloads\final_dataset (train+val)\train\GT'
folder_path3 = r'C:\Users\rajat\Downloads\final_dataset (train+val)\val\hazy'
folder_path4  = r'C:\Users\rajat\Downloads\final_dataset (train+val)\val\GT'


epochs = 30



def load(hazy_file, GT_file):
    hazy_image = tf.io.read_file(hazy_file)
    GT_image = tf.io.read_file(GT_file)

    hazy_image = tf.image.decode_jpeg(hazy_image, channels=3)
    GT_image = tf.image.decode_jpeg(GT_image, channels=3)

    hazy_image = tf.cast(hazy_image, tf.float32)
    GT_image = tf.cast(GT_image, tf.float32)
    return hazy_image, GT_image


x,y = load(os.path.join(pathHazy, "1.png"),os.path.join(pathGT, "1.png"))



def normalize(hazy_image, GT_image):
    hazy_image = (hazy_image / 127.5) - 1
    GT_image = (GT_image / 127.5) - 1
    return hazy_image, GT_image

def resize(hazy_image, GT_image):
    hazy_image = tf.image.resize(hazy_image, [IMAGE_SIZE, IMAGE_SIZE], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    GT_image = tf.image.resize(GT_image, [IMAGE_SIZE, IMAGE_SIZE], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return hazy_image, GT_image

def random_crop(hazy_image, GT_image):
    stacked_image = tf.stack([hazy_image, GT_image], axis = 0)
    cropped_image = tf.image.random_crop(stacked_image, size = [2, IMAGE_SIZE, IMAGE_SIZE, 3])
    return cropped_image[0], cropped_image[1]


def random_jitter(hazy_image, GT_image):
    if tf.random.uniform(()) > 0.5:
        hazy_image = tf.image.flip_left_right(hazy_image)
        GT_image = tf.image.flip_left_right(GT_image)
    return hazy_image, GT_image

def load_train_images(hazy_file, GT_file):
    hazy_image1, GT_image1 = load(hazy_file, GT_file)
    hazy_image2, GT_image2 = resize(hazy_image1, GT_image1)
    hazy_image3, GT_image3 = random_jitter(hazy_image2, GT_image2)
    hazy_image4, GT_image4 = normalize(hazy_image3, GT_image3)
    return hazy_image4, GT_image4

def load_test_image(hazy_file, GT_file):
    hazy_image1, GT_image1 = load(hazy_file, GT_file)
    hazy_image2, GT_image2 = resize(hazy_image1, GT_image1)
    hazy_image3, GT_image3 = normalize(hazy_image2, GT_image2)
    return hazy_image3, GT_image3


def list_files_in_folder(folder_path):
    file_locations = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_location = os.path.join(root, file_name)
            file_locations.append(file_location)
    return file_locations

file_locationsGT = list_files_in_folder(pathGT)

file_locationshazy = list_files_in_folder(pathHazy)

train_dataset = tf.data.Dataset.from_tensor_slices((file_locationshazy, file_locationsGT))

train_dataset = train_dataset.map(load_train_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(10).batch(BATCH_SIZE)


num_examples = train_dataset.reduce(tf.constant(0), lambda x, _: x + 1).numpy()
print("Number of examples in train_dataset:", num_examples)




file_locationsGTv = list_files_in_folder(folder_path3)


file_locationshazyv = list_files_in_folder(folder_path4)

test_dataset = tf.data.Dataset.from_tensor_slices((file_locationshazyv, file_locationsGTv))

test_dataset = test_dataset.map(load_test_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.shuffle(10).batch(BATCH_SIZE)


# downsample block
def downsample(filters, size, batchnorm=True):
    init = tf.random_normal_initializer(0., 0.02)
    result = Sequential()
    result.add(Conv2D(filters, size, strides=2, padding="same", kernel_initializer=init, use_bias=False))
    if batchnorm == True:
        result.add(BatchNormalization())

    result.add(LeakyReLU())
    return result


down_model = downsample(3, 4)
down_result = down_model(tf.expand_dims(x, axis=0))
print(down_result.shape)


# upsample block
def upsample(filters, size, dropout = False):
    init = tf.random_normal_initializer(0, 0.02)
    result = Sequential()
    result.add(Conv2DTranspose(filters, size, strides = 2, padding = "same", kernel_initializer = init, use_bias = False))
    result.add(BatchNormalization())
    if dropout == True:
        result.add(Dropout(0.5))
    result.add(ReLU())
    return result
up_model = upsample(3,4)
up_result = up_model(down_result)
print(up_result.shape)


def generator():
    inputs = Input(shape=[IMAGE_SIZE, IMAGE_SIZE, 3])
    down_stack = [
        downsample(64, 4, batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4)
    ]

    up_stack = [
        upsample(512, 4, dropout=True),
        upsample(512, 4, dropout=True),
        upsample(512, 4),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]
    init = tf.random_normal_initializer(0., 0.02)
    last = Conv2DTranspose(3, 4, strides=2, padding="same", kernel_initializer=init, activation="tanh")
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = Concatenate()([x, skip])

    x = last(x)
    return Model(inputs=inputs, outputs=x)

gen = generator()
gen.summary()



LAMBDA = 100
from keras.losses import BinaryCrossentropy
loss_function = BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_function(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


def discriminator():
    init = tf.random_normal_initializer(0., 0.02)

    inp = Input(shape=[IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    tar = Input(shape=[IMAGE_SIZE, IMAGE_SIZE, 3], name="target_image")
    x = Concatenate()([inp, tar])
    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)

    zero_pad1 = ZeroPadding2D()(down3)
    conv = Conv2D(256, 4, strides=1, kernel_initializer=init, use_bias=False)(zero_pad1)
    leaky_relu = LeakyReLU()(conv)
    zero_pad2 = ZeroPadding2D()(leaky_relu)
    last = Conv2D(1, 4, strides=1, kernel_initializer=init)(zero_pad2)
    return Model(inputs=[inp, tar], outputs=last)

disc = discriminator()
disc.summary()



def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_function(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_function(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

generator_optimizer = Adam(learning_rate=2e-4, beta_1=0.5)
discriminator_optimizer = Adam(learning_rate=2e-4, beta_1=0.5)

def save_images(model, test_input, target, epoch):
    prediction = model(test_input, training= False)
    plt.figure(figsize = (15,15))
    display_list= [test_input[0], target[0], prediction[0]]
    title = ["Input Image", "Ground Truth", "Predicton Image"]
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis("off")
    plt.savefig(f"output/epoch_{epoch}.jpg")
    plt.close()





@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = gen(input_image, training=True)

        disc_real_output = disc([input_image, target], training=True)
        disc_generated_output = disc([input_image, gen_output], training=True)
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
        generator_gradients = gen_tape.gradient(gen_total_loss, gen.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, disc.trainable_variables)
        generator_optimizer.apply_gradients(zip(generator_gradients, gen.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients, disc.trainable_variables))
        return gen_total_loss, disc_loss


def fit(train_ds, epochs, test_ds):
    for epoch in range(epochs):
        start = time.time()
        for input_, target in test_ds.take(1):
            save_images(gen, input_, target, epoch)
        # Train
        print(f"Epoch {epoch}")
        for n, (input_, target) in train_ds.enumerate():
            gen_loss, disc_loss = train_step(input_, target, epoch)
        print("Generator loss {:.2f} Discriminator loss {:.2f}".format(gen_loss, disc_loss))
        print("Time take for epoch {} is {} sec\n".format(epoch + 1, time.time() - start))


keras.backend.clear_session()

fit(train_dataset, epochs, test_dataset)

gen.save('generator.h5')
disc.save('discriminator.h5')