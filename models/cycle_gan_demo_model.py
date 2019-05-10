import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

class CycleGANDemoModel(BaseModel):
    def name(self):
        return 'CycleGANDemoModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        # self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['fake']
        self.generator_type = opt.which_direction
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if (self.generator_type == 'AtoB'):
            self.model_names = ['G_A']
        else:
            self.model_names = ['G_B']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        if (self.generator_type == 'AtoB'):
            self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        else:
            self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

    def set_input(self, input):
        self.real = input['img'].to(self.device)
        self.mask = input['mask'].to(self.device)
        # print (self.real.size())
        # print (self.mask.size())

    def mask_img(self, img, mask):
        img_denormal = (img + 1) / 2
        result_denormal = torch.mul(img_denormal,mask)
        result_img = (result_denormal - 0.5) * 2
        return result_img

    def forward(self):
        self.real = self.mask_img(self.real, self.mask)
        if (self.generator_type == 'AtoB'):
            self.fake = self.netG_A(self.real)
        else:
            self.fake = self.netG_B(self.real)
        self.fake = self.mask_img(self.fake,self.mask)

