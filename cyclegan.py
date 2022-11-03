import numpy as np
import cv2
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (
    Activation,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
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
        """_summary_

        Args:
            height (_type_): _description_
            width (_type_): _description_
        """
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
        """_summary_

        Args:
            image_pool (_type_): _description_
            images (_type_): _description_
            max_size (int, optional): _description_. Defaults to 50.

        Returns:
            _type_: _description_
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
        """_summary_"""
        if not os.path.exists("models"):
            os.makedirs("models")
        self.g_model_AB.save("models/g_model_AB.h5")
        self.g_model_BA.save("models/g_model_BA.h5")
        self.d_model_A.save("models/d_model_A.h5")
        self.d_model_B.save("models/d_model_B.h5")

    def test_model(self, n):
        X_real_A, _ = self.data_generator.generate_real_samples(1, self.d_model_A.output_shape[1])
        X_mask_B, _ = self.data_generator.generate_mask_samples(1, self.d_model_A.output_shape[1])
        X_real_A2B = self.g_model_AB.predict(X_real_A)
        X_mask_B2A = self.g_model_BA.predict(X_mask_B)
        X_real_A2B2A = self.g_model_BA.predict(X_real_A2B)
        X_real_B2A2B = self.g_model_AB.predict(X_mask_B2A)

        plt.figure(figsize=(20, 10))
        plt.subplot(2, 3, 1)
        plt.imshow(np.squeeze(X_real_A) * -1, cmap="gist_gray")
        plt.axis(False)
        plt.subplot(2, 3, 2)
        plt.imshow(np.squeeze(X_real_A2B))
        plt.axis(False)
        plt.subplot(2, 3, 3)
        plt.imshow(np.squeeze(X_real_A2B2A) * -1, cmap="gist_gray")
        plt.axis(False)

        plt.subplot(2, 3, 4)
        plt.imshow(np.squeeze(X_mask_B))
        plt.axis(False)
        plt.subplot(2, 3, 5)
        plt.imshow(np.squeeze(X_mask_B2A) * -1, cmap="gist_gray")
        plt.axis(False)
        plt.subplot(2, 3, 6)
        plt.imshow(np.squeeze(X_real_B2A2B))
        plt.axis(False)

        plt.savefig("models/best_g_model_epoch_%0.6d.png" % n)
        plt.close()

    def train(self, n_epochs, n_batch):
        """_summary_

        Args:
            n_epochs (_type_): _description_
            n_batch (_type_): _description_

        Raises:
            ValueError: _description_
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
        self.n_filt = n_filt
        self.height = height
        self.width = width
        self.input_shape = (self.height, self.width, 1)

    def build(self):
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
        self.n_filt = n_filt
        self.n_resnet_layers = n_resnet_layers
        self.height = height
        self.width = width
        self.input_shape = (self.height, self.width, 1)

    def _resnet_block(self, n_filt, input_layer):
        # weight initialisation
        init = RandomNormal(stddev=0.2, seed=1)

        r = Conv2D(n_filt, (3, 3), padding="same", kernel_initializer=init)(input_layer)
        r = InstanceNormalization(axis=-1)(r)
        r = Activation("relu")(r)

        r = Conv2D(n_filt, (3, 3), padding="same", kernel_initializer=init)(r)
        r = InstanceNormalization(axis=-1)(r)

        return Concatenate()([r, input_layer])

    def upsample_block(self, n_filt, input_layer):
        x = UpSampling2D(size=(2, 2))(input_layer)
        x = Conv2D(n_filt, (3, 3), padding='same')(x)

        return x

    def build(self):
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
    def __init__(self, height, width, g_model1, g_model2, d_model) -> None:
        self.height = height
        self.width = width
        self.input_shape = (self.height, self.width, 1)
        self.g_model1 = g_model1
        self.g_model2 = g_model2
        self.d_model = d_model

    def build(self):
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
    def __init__(self, filenames, height, width):
        """_summary_

        Args:
            filenames (_type_): _description_
            height (_type_): _description_
            width (_type_): _description_
        """
        self.filenames = filenames
        self.images = None
        self.height = height
        self.width = width
        self.seq = iaa.Sequential([iaa.Fliplr(0.5), iaa.Flipud(0.5)])

        self.load_images()

    def normalise_image(self, image):
        """_summary_

        Args:
            image (_type_): _description_

        Returns:
            _type_: _description_
        """

        return ((image - np.amin(image)) / (np.amax(image) - np.amin(image))) * 2 - 1

    def load_images(self):
        """_summary_"""
        self.images = []
        for f in self.filenames:
            img = cv2.imread(f, -1)
            img = cv2.resize(img, (self.height, self.width), interpolation=cv2.INTER_AREA)
            img = 255 - cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = self.normalise_image(img)
            self.images += [img]

    def generate_crack(self):
        """_summary_

        Returns:
            _type_: _description_
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

        t = np.random.randint(1, 5)
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
        """_summary_

        Args:
            n_samples (_type_): _description_
            patch_shape (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
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
        """_summary_

        Args:
            n_samples (_type_): _description_
            patch_shape (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        X = np.zeros((n_samples, self.height, self.width, 1))

        for i in range(n_samples):
            img = self.normalise_image(self.generate_crack())
            X[i, :, :, 0] = self.seq(images=img)

        Y = np.ones((n_samples, patch_shape, patch_shape, 1))

        return X, Y

    def generate_fake_samples(self, g_model, AB, patch_shape=None):
        X = g_model.predict(AB, verbose=0)
        Y = np.zeros((len(X), patch_shape, patch_shape, 1))

        return X, Y
