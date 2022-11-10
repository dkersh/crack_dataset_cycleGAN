import cyclegan
import argparse
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('-n_epochs', type=int, default=1000)
parser.add_argument('-batch_size', type=int, default=4)
parser.add_argument('-img_height', type=int, default=128)
parser.add_argument('-img_width', type=int, default=128)

args = parser.parse_args()

n_epochs = args.n_epochs
batch_size = args.batch_size
img_height = args.img_height
img_width = args.img_width

filenames = glob('crack_segmentation_dataset/images/DeepCrack*.jpg')

cyclegan_model = cyclegan.CycleGAN(img_height, img_width)
cyclegan_model.data_generator = cyclegan.DataGenerator(filenames, 128, 128)

cyclegan_model.train(n_epochs, batch_size)