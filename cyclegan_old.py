from glob import glob

import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Activation, BatchNormalization,
                                     Concatenate, Conv2D, Conv2DTranspose,
                                     Dense, Dropout, Flatten, Input, LeakyReLU,
                                     Reshape, ZeroPadding2D)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow_addons.layers import InstanceNormalization


class cycleGAN():

    def __init__(self, N_channels = 1, input_width = 128, input_height = 128, dataset_name = None):

        self.N_channels = N_channels
        self.input_width = input_width
        self.input_height = input_height
        self.dataset_name = dataset_name
        self.image_shape = (self.input_width, self.input_height, self.N_channels)

        self.image_poolA = list()
        self.image_poolB = list()

        self.dataset_A = None
        self.dataset_B = None

        self.dataset = None
        
        self.directory = ''

        self.g_model_AB = self.define_generator(128)
        self.g_model_BA = self.define_generator(128)
        self.d_model_A = self.define_discriminator(128)
        self.d_model_B = self.define_discriminator(128)

        self.c_model_AB = self.composite_model(self.g_model_AB, self.d_model_B, self.g_model_BA)
        self.c_model_BA = self.composite_model(self.g_model_BA, self.d_model_A, self.g_model_AB)

    # Define Discriminator
    def define_discriminator(self, n_filt):
        # weight initialisation
        init = RandomNormal(stddev=0.02, seed=1)
        input_img = Input(shape=self.image_shape)

        #c1
        c1 = Conv2D(n_filt, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(input_img)
        c1 = LeakyReLU(alpha=0.2)(c1)
        #c2
        c2 = Conv2D(n_filt*2, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(c1)
        c2 = InstanceNormalization(axis=-1)(c2)
        c2 = LeakyReLU(alpha=0.2)(c2)
        #c3
        c3 = Conv2D(n_filt*4, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(c2)
        c3 = InstanceNormalization(axis=-1)(c3)
        c3 = LeakyReLU(alpha=0.2)(c3)
        #c4
        c4 = Conv2D(n_filt*8, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(c3)
        c4 = InstanceNormalization(axis=-1)(c4)
        c4 = LeakyReLU(alpha=0.2)(c4)
        #c5
        c5 = Conv2D(n_filt*8, (4, 4), padding='same', kernel_initializer=init)(c4)
        c5 = InstanceNormalization(axis=-1)(c5)
        c5 = LeakyReLU(alpha=0.2)(c5)
        # Patch Output
        patch_out = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(c5)
        # Define model
        model = Model(input_img, patch_out)
        model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])

        return model

    def define_generator(self, n_filt, n_resnet_layers=6):
        #weight initialisation
        init = RandomNormal(stddev=0.02, seed=1)
        input_img = Input(shape=self.image_shape)
        # c1
        c1 = Conv2D(n_filt, (7, 7), padding='same', kernel_initializer=init)(input_img)
        c1 = InstanceNormalization(axis=-1)(c1)
        c1 = Activation('relu')(c1)
        # c2
        c2 = Conv2D(n_filt*2, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(c1)
        c2 = InstanceNormalization(axis=-1)(c2)
        c2 = Activation('relu')(c2)
        # c3
        c3 = Conv2D(n_filt*4, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(c2)
        c3 = InstanceNormalization(axis=-1)(c3)
        r = Activation('relu')(c3)
        # ResNet Blocks
        for _ in range(n_resnet_layers):
            r = self.resnet_block(n_filt*4, r)
        #u1
        u1 = Conv2DTranspose(n_filt*2, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(r)
        u1 = InstanceNormalization(axis=-1)(u1)
        u1 = Activation('relu')(u1)
        #u2
        u2 = Conv2DTranspose(n_filt, (3, 3), strides=(2, 2), padding='same', kernel_initializer=init)(u1)
        u2 = InstanceNormalization(axis=-1)(u2)
        u2 = Activation('relu')(u2)
        #u3
        u3 = Conv2D(self.N_channels, (7, 7), padding='same', kernel_initializer=init)(u2)
        u3 = InstanceNormalization(axis=-1)(u3)
        output = Activation('tanh')(u3)

        model = Model(input_img, output)
        return model

    def composite_model(self, g_model1, d_model, g_model2):
        g_model1.trainable = True
        d_model.trainable = False
        g_model2.trainable = False

        # Discriminator Element
        input_gen = Input(shape=self.image_shape)
        g_model1_output = g_model1(input_gen)
        d_model_output = d_model(g_model1_output)
        # Identity Element
        input_id = Input(shape=self.image_shape)
        output_id = g_model1(input_id)
        # Forward cycle
        output_f = g_model2(g_model1_output)
        # Backward cycle
        g_model2_output = g_model2(input_id)
        output_b = g_model1(g_model2_output)

        model = Model([input_gen, input_id], [d_model_output, output_id, output_f, output_b])
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)

        return model

    def resnet_block(self, n_filt, input_layer):
        #weight initialisation
        init = RandomNormal(stddev=0.2, seed=1)
        
        r = Conv2D(n_filt, (3, 3), padding='same', kernel_initializer=init)(input_layer)
        r = InstanceNormalization(axis=-1)(r)
        r = Activation('relu')(r)

        r = Conv2D(n_filt, (3, 3), padding='same', kernel_initializer=init)(r)
        r = InstanceNormalization(axis=-1)(r)

        return Concatenate()([r, input_layer])

    def normaliseImg(self, img):

        img_norm = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

        return (img_norm * 2) - 1

    def normaliseImgBack(self, img):

        return (img+1)/2
    
    def getRandomCrop(self, img, crop_height, crop_width):
        max_x = np.shape(img)[1] - crop_width
        max_y = np.shape(img)[0] - crop_height

        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)

        crop = img[y:y+crop_height, x:x+crop_width]

        # Apply augmentation to sample
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.LinearContrast((0.75, 1.5)),
            iaa.Multiply((0.8, 1.2))
        ])

        crop = seq.augment_image(crop)

        return np.expand_dims(crop, axis=-1)
    
    def generateRandomMask(self):
        # Blank canvas
        img = np.zeros((128, 128))

        # Random parameters
        nCircles = np.random.randint(20, 60)
        #nCircles = int(np.ceil(np.random.normal(15, 10, 1)))
        #print(nCircles)

        for i in range(nCircles):
            #R = np.random.randint(5,12)
            x = np.random.randint(0, 128)
            y = np.random.randint(0, 128)

            #img = cv2.circle(img, (x, y), R, color=255, thickness=-1)
            R1 = np.random.randint(5, 15)
            R2 = np.random.randint(R1-3, R1+3)
            angle = np.random.randint(0, 360)
            startAngle = 0
            endAngle = 360
            img = cv2.ellipse(img, (x, y), (R1, R2), angle, startAngle, endAngle, color=1, thickness=-1)

        return np.expand_dims(img, axis=-1)
    
    def generateRandomMaskNoOverlap(self):
        # Random parameters
        nCircles = np.random.randint(20, 60)

        # Blank array
        img = np.zeros((nCircles, 128, 128))
        imgB = np.zeros((nCircles, 128, 128))
        Rmag_arr = np.zeros((nCircles))

        img2 = np.zeros((128, 128))

        for i in range(nCircles):
            #R = np.random.randint(5,12)
            x = np.random.randint(0, 128)
            y = np.random.randint(0, 128)

            #img = cv2.circle(img, (x, y), R, color=255, thickness=-1)
            R1 = np.random.randint(5, 15)
            R2 = np.random.randint(R1-3, R1+3)
            Rmag = np.sqrt(R1**2 + R2**2)
            angle = np.random.randint(0, 360)
            startAngle = 0
            endAngle = 360
            img[i] = cv2.ellipse(img[i], (x, y), (R1, R2), angle, startAngle, endAngle, color=1, thickness=-1)
            imgB[i] = cv2.ellipse(imgB[i], (x, y), (R1+3, R2+3), angle, startAngle, endAngle, color=1, thickness=-1)
            Rmag_arr[i] = Rmag

            img2 = img2 + img[i]

            if np.amax(img2) > 1:
                img2 = img2 + imgB[i]
                img2 = img2 + imgB[i]
                img2[img2 > 1] = 0
                img2 = img2 + img[i]



        return np.expand_dims(img2, axis=-1)

    def generate_real_samples(self, n_samples, patch_shape):

        X = np.zeros((n_samples, self.input_width, self.input_height, self.N_channels))
        
        img = cv2.imread(FILENAME HERE, -1)[:,:,0]

        for i in range(n_samples):
           
            X[i] = self.normaliseImg(self.getRandomCrop(img, 128, 128))

        Y = np.ones((n_samples, patch_shape, patch_shape, 1)) # Labels for real images

        return X, Y
    
    def generate_mask_samples(self, n_samples, patch_shape):
        
        X = np.zeros((n_samples, self.input_width, self.input_height, self.N_channels))
        
        for i in range(n_samples):
            X[i] = self.normaliseImg(self.generateRandomMaskNoOverlap())
           
        Y = np.ones((n_samples, patch_shape, patch_shape, 1))
        
        return X, Y

    def generate_fake_samples(self, g_model, dataset, patch_shape):
        # Generate fake images
        X = g_model.predict(dataset)

        # These are fake images, so are represented as 0s
        Y = np.zeros((len(X), patch_shape, patch_shape, 1))

        return X, Y

    def update_image_pool(self, img_pool, images, max_size=50):
        selected = list()

        for image in images:
            if len(img_pool) < max_size:
                # Add images to the pool
                img_pool.append(image)
                selected.append(image)
            elif np.random.uniform(0, 1, 1) < 0.5:
                # If pool full, either use a new image
                selected.append(image)
            else:
                # Or replace an existing image and use replacement
                ix = np.random.randint(0, len(img_pool))
                selected.append(img_pool[ix])
                img_pool[ix] = image

        return np.array(selected)

    def setDatasetA_path(self, path):
        filenames = np.array(glob(path))

        self.dataset_A = filenames

    def setDatasetB_path(self, path):
        filenames = np.array(glob(path))

        self.dataset_B = filenames

    def testCycleGAN(self, epoch):
        plt.figure(figsize=(40, 20))
        for i in range(10):
            X_real_A, _ = self.generate_real_samples(1, self.d_model_A.output_shape[1])
            X_real_B, _ = self.generate_mask_samples(1, self.d_model_A.output_shape[1])

            pred_rAfB = self.g_model_AB.predict(X_real_A)
            pred_rBfA = self.g_model_BA.predict(X_real_B)
            pred_fArB = self.g_model_AB.predict(pred_rBfA)
            pred_fBaA = self.g_model_BA.predict(pred_rAfB)

            X_real_A = self.normaliseImgBack(X_real_A)
            X_real_B = self.normaliseImgBack(X_real_B)
            pred_rAfB = self.normaliseImgBack(pred_rAfB)
            pred_rBfA = self.normaliseImgBack(pred_rBfA)
            pred_fArB = self.normaliseImgBack(pred_fArB)
            pred_fBaA = self.normaliseImgBack(pred_fBaA)

            plt.subplot(6, 10, i+1)
            plt.imshow(np.squeeze(X_real_A))
            plt.subplot(6, 10, i+1+10)
            plt.imshow(np.squeeze(pred_rAfB))
            plt.subplot(6, 10, i+1+20)
            plt.imshow(np.squeeze(pred_fBaA))
            plt.subplot(6, 10, i+1+30)
            plt.imshow(np.squeeze(X_real_B))
            plt.subplot(6, 10, i+1+40)
            plt.imshow(np.squeeze(pred_rBfA))
            plt.subplot(6, 10, i+1+50)
            plt.imshow(np.squeeze(pred_fArB))
            
        filename = 'epoch_%0.6d.png' % (epoch)
        
        plt.savefig(self.directory + filename)
        plt.close()
        
    def saveModels(self, epoch):
        self.g_model_AB.save(self.directory + 'g_model_AB.h5')
        self.g_model_BA.save(self.directory + 'g_model_BA.h5')
        self.d_model_A.save(self.directory + 'd_model_A.h5')
        self.d_model_B.save(self.directory + 'd_model_B.h5')
        #self.c_model_AB.save('c_model_AB_epoch_%0.6d.h5' % (epoch))
        #self.c_model_BA.save('c_model_BA_epoch_%0.6d.h5' % (epoch))

    def train(self, n_epochs, n_batch):
        
        n_patch = self.d_model_A.output_shape[1]
        #n_steps = int(len(self.dataset_A) / n_batch)
        n_steps = 100

        n = 1

        for i in range(n_epochs):
            for j in range(n_steps):
                X_real_A, Y_real_A = self.generate_real_samples(n_batch, self.d_model_A.output_shape[1])
                X_real_B, Y_real_B = self.generate_mask_samples(n_batch, self.d_model_A.output_shape[1])
                
                # Generate fake samples
                X_fake_A, Y_fake_A = self.generate_fake_samples(self.g_model_BA, X_real_B, n_patch)
                X_fake_B, Y_fake_B = self.generate_fake_samples(self.g_model_AB, X_real_A, n_patch)

                # Update fakes from image pool
                X_fake_A = self.update_image_pool(self.image_poolA, X_fake_A)
                X_fake_B = self.update_image_pool(self.image_poolB, X_fake_B)

                # Update generator B->A via adversarial and cycle loss
                g_loss2, _, _, _, _ = self.c_model_BA.train_on_batch([X_real_B, X_real_A], [Y_real_A, X_real_A, X_real_B, X_real_A])
                dA_loss1 = self.d_model_A.train_on_batch(X_real_A, Y_real_A)
                dA_loss2 = self.d_model_A.train_on_batch(X_fake_A, Y_fake_A)
                g_loss1, _, _, _, _ = self.c_model_AB.train_on_batch([X_real_A, X_real_B], [Y_real_B, X_real_B, X_real_A, X_real_B])
                dB_loss1 = self.d_model_B.train_on_batch(X_real_B, Y_real_B)
                dB_loss2 = self.d_model_B.train_on_batch(X_fake_B, Y_fake_B)

                print('>%d / %d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (n, n_epochs*n_steps, dA_loss1, dA_loss2, dB_loss1, dB_loss2, g_loss1, g_loss2))

                if np.mod(n, int((n_epochs*n_steps)/100)) == 0:
                    self.testCycleGAN(n)
                    self.saveModels(n)
            
                n += 1
