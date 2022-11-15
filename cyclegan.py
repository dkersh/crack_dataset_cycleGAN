import numpy as np
import cv2
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (
    Activation,
    Concatenate,
    Conv2D,
    Input,
    LeakyReLU,
    UpSampling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import InstanceNormalization
import os
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt


class CycleGAN:
    def __init__(self, height, width) -> None:
        """One channel image-based CycleGAN neural network model.

        Args:
            height (int): Input image height
            width (int): Input image width.
        """
        self.first_test = True
        self.test_data = None
        self.height = height
        self.width = width
        self.data_generator = None
        self.image_poolA = list()
        self.image_poolB = list()

        self.g_model_AB = Generator(self.height, self.width, 128).build()
        self.g_model_BA = Generator(self.height, self.width, 128).build()
        self.d_model_A = Discriminator(self.height, self.width, 128).build()
        self.d_model_B = Discriminator(self.height, self.width, 128).build()
        self.c_model_AB = CompositeModel(
            self.height, self.width, self.g_model_AB, self.g_model_BA, self.d_model_B
        ).build()
        self.c_model_BA = CompositeModel(
            self.height, self.width, self.g_model_BA, self.g_model_AB, self.d_model_A
        ).build()

    def update_image_pool(self, image_pool, images, max_size=50):
        """Image pooling used to assist with model training. List of previously used images which is updated every epoch.

        Args:
            image_pool (list): Image pool list
            images (numpy array): Images to update the image pool
            max_size (int, optional): Maximum size of the image pool. Defaults to 50.

        Returns:
            numpy array: Selected images for training.
        """
        selected = list()

        for image in images:
            if len(image_pool) < max_size:
                # Add images to the pool
                image_pool.append(image)
                selected.append(image)
            elif np.random.uniform(0, 1, 1) < 0.5:
                # If pool full, either use a new image
                selected.append(image)
            else:
                # Or replace an existing image and use replacement
                ix = np.random.randint(0, len(image_pool))
                selected.append(image_pool[ix])
                image_pool[ix] = image

        return np.array(selected)

    def save_model(self):
        """Save Models to file."""

        if not os.path.exists("models"):
            os.makedirs("models")
        self.g_model_AB.save("models/g_model_AB.h5")
        self.g_model_BA.save("models/g_model_BA.h5")
        self.d_model_A.save("models/d_model_A.h5")
        self.d_model_B.save("models/d_model_B.h5")

    def test_model(self, n):
        """Test the model with the same images every time a new minimum loss is found. Save plot to file.

        Args:
            n (int): epoch number
        """
        if self.first_test == True:
            X_real_A, _ = self.data_generator.generate_real_samples(4, self.d_model_A.output_shape[1])
            X_mask_B, _ = self.data_generator.generate_mask_samples(4, self.d_model_A.output_shape[1])
            self.first_test = False
            self.test_data = [X_real_A, X_mask_B]
        X_real_A, X_mask_B = self.test_data 
        X_real_A2B = self.g_model_AB.predict(X_real_A)
        X_mask_B2A = self.g_model_BA.predict(X_mask_B)
        X_real_A2B2A = self.g_model_BA.predict(X_real_A2B)
        X_real_B2A2B = self.g_model_AB.predict(X_mask_B2A)

        plt.figure(figsize=(10, 20))
        for i in range(len(X_real_A)):
            plt.subplot(8, 3, 1+(i*3))
            plt.imshow(np.squeeze(X_real_A[i]) * -1, cmap="gist_gray")
            plt.axis(False)
            plt.subplot(8, 3, 2+(i*3))
            plt.imshow(np.squeeze(X_real_A2B[i]))
            plt.axis(False)
            plt.subplot(8, 3, 3+(i*3))
            plt.imshow(np.squeeze(X_real_A2B2A[i]) * -1, cmap="gist_gray")
            plt.axis(False)

        for i in range(len(X_mask_B)):
            plt.subplot(8, 3, 13+(i*3))
            plt.imshow(np.squeeze(X_mask_B[i]))
            plt.axis(False)
            plt.subplot(8, 3, 14+(i*3))
            plt.imshow(np.squeeze(X_mask_B2A[i]) * -1, cmap="gist_gray")
            plt.axis(False)
            plt.subplot(8, 3, 15+(i*3))
            plt.imshow(np.squeeze(X_real_B2A2B[i]))
            plt.axis(False)

        plt.savefig("models/best_g_model_epoch_%0.6d.png" % n)
        plt.close()

    def train(self, n_epochs, n_batch):
        """Model train model.

        Args:
            n_epochs (int): Number of training epochs.
            n_batch (int): Batch_size

        Raises:
            ValueError: Raise error if no data generator has been allocated.
        """
        if self.data_generator == None:
            raise ValueError("Please allocate a data generator")
        n_patch = self.d_model_A.output_shape[1]
        n_steps = 100

        n = 1

        best_g_loss1 = 100

        for _ in range(n_epochs):
            for _ in range(n_steps):
                X_real_A, Y_real_A = self.data_generator.generate_real_samples(n_batch, self.d_model_A.output_shape[1])
                X_real_B, Y_real_B = self.data_generator.generate_mask_samples(n_batch, self.d_model_A.output_shape[1])

                # Generate fake samples
                X_fake_A, Y_fake_A = self.data_generator.generate_fake_samples(self.g_model_BA, X_real_B, n_patch)
                X_fake_B, Y_fake_B = self.data_generator.generate_fake_samples(self.g_model_AB, X_real_A, n_patch)

                # Update fakes from image pool
                X_fake_A = self.update_image_pool(self.image_poolA, X_fake_A)
                X_fake_B = self.update_image_pool(self.image_poolB, X_fake_B)

                # Update generator B->A via adversarial and cycle loss
                g_loss2, _, _, _, _ = self.c_model_BA.train_on_batch(
                    [X_real_B, X_real_A], [Y_real_A, X_real_A, X_real_B, X_real_A]
                )
                dA_loss1 = self.d_model_A.train_on_batch(X_real_A, Y_real_A)
                dA_loss2 = self.d_model_A.train_on_batch(X_fake_A, Y_fake_A)
                g_loss1, _, _, _, _ = self.c_model_AB.train_on_batch(
                    [X_real_A, X_real_B], [Y_real_B, X_real_B, X_real_A, X_real_B]
                )
                dB_loss1 = self.d_model_B.train_on_batch(X_real_B, Y_real_B)
                dB_loss2 = self.d_model_B.train_on_batch(X_fake_B, Y_fake_B)

                print(
                    ">%d / %d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]"
                    % (n, n_epochs * n_steps, dA_loss1, dA_loss2, dB_loss1, dB_loss2, g_loss1, g_loss2)
                )

                n += 1

                if (n > 1000) and (g_loss1 < best_g_loss1):
                    self.test_model(n)
                    best_g_loss1 = g_loss1
                    print("Saving Model...")
                    self.save_model()


class Discriminator:
    def __init__(self, height, width, n_filt) -> None:
        """CycleGAN Discriminator model. Used to distinguishing between real and fake samples.

        Args:
            height (int): Image height.
            width (int): Image width.
            n_filt (int): Base number of Convolutional filters.
        """
        self.n_filt = n_filt
        self.height = height
        self.width = width
        self.input_shape = (self.height, self.width, 1)

    def build(self):
        """Build discriminator model.

        Returns:
            model: Return compiled keras model.
        """
        # weight initialisation
        init = RandomNormal(stddev=0.02, seed=1)
        input_img = Input(shape=self.input_shape)

        # c1
        c1 = Conv2D(self.n_filt, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(input_img)
        c1 = LeakyReLU(alpha=0.2)(c1)
        # c2
        c2 = Conv2D(self.n_filt * 2, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(c1)
        c2 = InstanceNormalization(axis=-1)(c2)
        c2 = LeakyReLU(alpha=0.2)(c2)
        # c3
        c3 = Conv2D(self.n_filt * 4, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(c2)
        c3 = InstanceNormalization(axis=-1)(c3)
        c3 = LeakyReLU(alpha=0.2)(c3)
        # c4
        c4 = Conv2D(self.n_filt * 8,(4, 4),strides=(2, 2),padding="same", kernel_initializer=init)(c3)
        c4 = InstanceNormalization(axis=-1)(c4)
        c4 = LeakyReLU(alpha=0.2)(c4)
        # c5
        c5 = Conv2D(self.n_filt * 8, (4, 4), padding="same", kernel_initializer=init)(c4)
        c5 = InstanceNormalization(axis=-1)(c5)
        c5 = LeakyReLU(alpha=0.2)(c5)
        # Patch Output
        patch_out = Conv2D(1, (4, 4), padding="same", kernel_initializer=init)(c5)
        # Define model
        model = Model(input_img, patch_out)
        model.compile(
            loss="mse",
            optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
            loss_weights=[0.5],
        )

        return model


class Generator:
    def __init__(self, height, width, n_filt, n_resnet_layers=6):
        """CycleGAN Generator model. Used to generator new real / fake samples.

        Args:
            height (int): Image height.
            width (int): Image width.
            n_filt (int): Base number of convolution filters.
            n_resnet_layers (int, optional): Number of ResNet layers used. Defaults to 6.
        """
        self.n_filt = n_filt
        self.n_resnet_layers = n_resnet_layers
        self.height = height
        self.width = width
        self.input_shape = (self.height, self.width, 1)

    def _resnet_block(self, n_filt, input_layer):
        """Resnet block as described in He et al. Deep Residual Learning for Image Recognition (2016)
        https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec

        Args:
            n_filt (int): Base number of convolution filters
            input_layer (keras layer): Input layer

        Returns:
            keras layer: residual layer
        """
        # weight initialisation
        init = RandomNormal(stddev=0.2, seed=1)

        r = Conv2D(n_filt, (3, 3), padding="same", kernel_initializer=init)(input_layer)
        r = InstanceNormalization(axis=-1)(r)
        r = Activation("relu")(r)

        r = Conv2D(n_filt, (3, 3), padding="same", kernel_initializer=init)(r)
        r = InstanceNormalization(axis=-1)(r)

        return Concatenate()([r, input_layer])

    def upsample_block(self, n_filt, input_layer):
        """Upsample block used to combat checkerboard artifacting in CycleGAN models.

        Args:
            n_filt (int): Base number of filters
            input_layer (keras layer): input layer

        Returns:
            keras layer: upsample block
        """
        x = UpSampling2D(size=(2, 2))(input_layer)
        x = Conv2D(n_filt, (3, 3), padding='same')(x)

        return x

    def build(self):
        """Build keras neural network model.

        Returns:
            keras model: keras generator model
        """
        # Weight initialisation
        init = RandomNormal(stddev=0.02, seed=1)
        input_img = Input(shape=self.input_shape)
        # c1
        c1 = Conv2D(self.n_filt, (7, 7), padding="same", kernel_initializer=init)(input_img)
        c1 = InstanceNormalization(axis=-1)(c1)
        c1 = Activation("relu")(c1)
        # c2
        c2 = Conv2D(self.n_filt * 2,(3, 3),strides=(2, 2),padding="same",kernel_initializer=init)(c1)
        c2 = InstanceNormalization(axis=-1)(c2)
        c2 = Activation("relu")(c2)
        # c3
        c3 = Conv2D(self.n_filt * 4,(3, 3),strides=(2, 2),padding="same",kernel_initializer=init)(c2)
        c3 = InstanceNormalization(axis=-1)(c3)
        r = Activation("relu")(c3)
        # ResNet Blocks
        for _ in range(self.n_resnet_layers):
            r = self._resnet_block(self.n_filt * 4, r)
        # u1
        #u1 = Conv2DTranspose(self.n_filt * 2, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(r)
        u1 = self.upsample_block(self.n_filt*2, r)
        u1 = InstanceNormalization(axis=-1)(u1)
        u1 = Activation("relu")(u1)
        # u2
        #u2 = Conv2DTranspose(self.n_filt, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(u1)
        u2 = self.upsample_block(self.n_filt, u1)
        u2 = InstanceNormalization(axis=-1)(u2)
        u2 = Activation("relu")(u2)
        # u3
        u3 = Conv2D(1, (7, 7), padding="same", kernel_initializer=init)(u2)
        u3 = InstanceNormalization(axis=-1)(u3)
        output = Activation("tanh")(u3)

        model = Model(input_img, output)
        return model


class CompositeModel:
    def __init__(self, height, width, g_model1, g_model2, d_model):
        """CycleGAN composite model. What is actually trained when training a CycleGAN model.

        Args:
            height (int): image height
            width (int): image width
            g_model1 (keras model): first generator model
            g_model2 (keras model): second generator model
            d_model (keras model): discriminator to distinguish real and fake samples.
        """
        self.height = height
        self.width = width
        self.input_shape = (self.height, self.width, 1)
        self.g_model1 = g_model1
        self.g_model2 = g_model2
        self.d_model = d_model

    def build(self):
        """Build composite CycleGAN model

        Returns:
            keras model: CycleGAN composite model
        """
        self.g_model1.trainable = True
        self.g_model2.trainable = False
        self.d_model.trainable = False

        # Discriminator Element
        input_gen = Input(shape=self.input_shape)
        g_model1_output = self.g_model1(input_gen)
        d_model_output = self.d_model(g_model1_output)
        # Identity Element
        input_id = Input(shape=self.input_shape)
        output_id = self.g_model1(input_id)
        # Forward cycle
        output_f = self.g_model2(g_model1_output)
        # Backward cycle
        g_model2_output = self.g_model2(input_id)
        output_b = self.g_model1(g_model2_output)

        model = Model([input_gen, input_id], [d_model_output, output_id, output_f, output_b])
        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss=["mse", "mae", "mae", "mae"], loss_weights=[1, 5, 10, 10], optimizer=opt)

        return model


class DataGenerator:
    def __init__(self, filenames, y_filenames, height, width):
        """Datagenerator class for providing samples for the CycleGAN mdoel

        Args:
            filenames (list): List of A domain images.
            y_filenames (list): List of B domain images
            height (int): Image height
            width (int): Image width
        """
        self.filenames = filenames
        self.y_filenames = y_filenames
        self.images = None
        self.real_cracks = None
        self.height = height
        self.width = width
        self.seq = iaa.Sequential([iaa.Fliplr(0.5), iaa.Flipud(0.5)])

        self.load_images()

    def normalise_image(self, image):
        """Normalise image between 1 and -1 for CycleGAN training.

        Args:
            image (numpy array): input image

        Returns:
            numpy array: Normalised image.
        """

        return ((image - np.amin(image)) / (np.amax(image) - np.amin(image))) * 2 - 1

    def load_images(self):
        """Load image from file and store in class attributes"""
        self.images = []
        for f in self.filenames:
            img = cv2.imread(f, -1)
            img = cv2.resize(img, (self.height, self.width), interpolation=cv2.INTER_AREA)
            img = 255 - cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = self.normalise_image(img)
            self.images += [img]

        self.real_cracks = []
        for f in self.y_filenames:
            img = cv2.imread(f, -1)
            img = cv2.resize(img, (self.height, self.width), interpolation=cv2.INTER_AREA)
            img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
            #img = img.astype(bool).astype(int)
            img = self.normalise_image(img)
            self.real_cracks += [img]
        

    def generate_crack(self):
        """Random walk algorithm for simulating a crack.

        Returns:
            numpy array: simulated crack image.
        """
        if np.random.randint(low=0, high=2) == 0:
            prev_point = [np.random.randint(low=0, high=self.height), 0]
            theta = np.random.randint(0, 180)
        else:
            prev_point = [0, np.random.randint(low=0, high=self.width)]
            theta = np.random.randint(-90, 90)

        crack_image = np.zeros((self.height, self.width))
        all_points = [prev_point]
        all_theta = [theta]

        n_cracks = 10
        n_vertices = 10
        crack_image = np.zeros((self.height, self.width))

        t = np.random.randint(1, 3)
        for _ in range(np.random.randint(1, 3)):
            for _ in range(n_cracks):
                for _ in range(n_vertices):
                    length = np.random.uniform(1, self.height // 5)
                    x2 = prev_point[0] + length * np.cos(theta * np.pi / 180)
                    y2 = prev_point[1] + length * np.sin(theta * np.pi / 180)
                    point = np.round([x2, y2]).astype(int)
                    crack_image = cv2.line(crack_image, prev_point, point, color=1, thickness=t)
                    prev_point = point
                    theta = np.random.uniform(low=theta - 45, high=theta + 45)
                    all_points += [prev_point]
                    all_theta += [theta]

                ind = np.random.randint(low=0, high=len(all_points))
                prev_point = all_points[ind]
                theta = all_theta[ind]

            crack_image += crack_image

        return crack_image.astype(bool).astype(int)

    def generate_real_samples(self, n_samples, patch_shape=None):
        """Method for generating real samples (i.e. real images of cracks)

        Args:
            n_samples (int): Number of samples
            patch_shape (int, optional): Patch shape as prescribed by PatchGAN. Defaults to None.

        Returns:
            numpy array, numpy array: array of samples, and array of labels (all ones because real)
        """

        X = np.zeros((n_samples, self.height, self.width, 1))

        for i in range(n_samples):
            ind = np.random.randint(0, len(self.images))
            img = self.images[ind]
            img = self.seq(images=img)

            X[i, :, :, 0] = img

        Y = np.ones((n_samples, patch_shape, patch_shape, 1))

        return X, Y

    def generate_mask_samples(self, n_samples, patch_shape=None):
        """Method for generating simulating cracks and crack segmentations.

        Args:
            n_samples (int): Number of samples
            patch_shape (int, optional): Patch shape as prescribed by PatchGAN. Defaults to None.

        Returns:
            numpy array, numpy array: array of samples, and array of labels (all ones because real)
        """

        X = np.zeros((n_samples, self.height, self.width, 1))

        for i in range(n_samples):
            if np.random.uniform(0, 1) > 0.5:
                img = self.generate_crack()
                img = self.normalise_image(img)
            else:
                ind = np.random.randint(0, len(self.real_cracks))
                img = self.real_cracks[ind]
                #img = img.astype(bool).astype(int)
            
            X[i, :, :, 0] = self.seq(images=img)

        Y = np.ones((n_samples, patch_shape, patch_shape, 1))

        return X, Y

    def generate_fake_samples(self, g_model, AB, patch_shape=None):
        """Method for generating fake samples using a generator model. Since they're generated with a model, they are
        considered fake i.e. synthetic.

        Args:
            g_model (keras model): Generator model
            AB (numpy array): input samples
            patch_shape (int, optional): Patch shape as prescribed by PatchGAN. Defaults to None.

        Returns:
            numpy array, numpy array: array of samples, and array of labels (all zeros because fake)
        """
        X = g_model.predict(AB, verbose=0)
        Y = np.zeros((len(X), patch_shape, patch_shape, 1))

        return X, Y
