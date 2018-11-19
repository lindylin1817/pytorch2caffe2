import os
import sys
import torch
import torch.onnx
import torch.utils.model_zoo
from torch.autograd import Variable
sys.path.append("../DeblurGAN")
from models.models import create_model
import models.networks as networks
from options.test_options import TestOptions
import shutil

batch_size = 1    # just a random number

# Load the pretrained model weights
model_path = './model/char_deblur/latest_net_G.pth'
state_dict = torch.utils.model_zoo.load_url(model_path, model_dir="./model/char_deblur")

# Load the DeblurnGAN neural network
gan_opt = TestOptions().parse()
gan_opt.name = "char_deblur"
gan_opt.checkpoints_dir = "./model/"
gan_opt.model = "test"
gan_opt.dataset_mode = "single"
gan_opt.dataroot = "/tmp/gan/"
try:
    shutil.rmtree(gan_opt.dataroot)
except:
    pass
os.mkdir(gan_opt.dataroot)
gan_opt.loadSizeX = 64
gan_opt.loadSizeY = 64
gan_opt.fineSize = 64
gan_opt.learn_residual = True
gan_opt.nThreads = 1  # test code only supports nThreads = 1
gan_opt.batchSize = 1  # test code only supports batchSize = 1
gan_opt.serial_batches = True  # no shuffle
gan_opt.no_flip = True  # no flip
#torch_model = create_model(gan_opt)

gpus = []
torch_model = networks.define_G(gan_opt.input_nc, gan_opt.output_nc, gan_opt.ngf,
                                gan_opt.which_model_netG, gan_opt.norm, not gan_opt.no_dropout, gpus, False,
                                gan_opt.learn_residual)

torch_model.load_state_dict(state_dict)
#torch_model.load_state_dict(state_dict)

# set the train mode to false since we will only run the forward pass.
torch_model.train(False)

# Input to the model
x = Variable(torch.randn(batch_size, 3, 60, 60), requires_grad=True)
x = x.float()

# Export the model
torch_out = torch.onnx._export(torch_model,             # model being run
                               x,                       # model input (or a tuple for multiple inputs)
                               "deblurring.onnx", # where to save the model (can be a file or file-like object)
                               verbose=False, export_params=False, training=False)      # store the trained parameter weights inside the model file