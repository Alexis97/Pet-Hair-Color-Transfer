from .base_options import BaseOptions


class DemoOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--which_epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')

        parser.set_defaults(model='cycle_gan_demo')
        parser.set_defaults(name='dog_mask_cyclegan_rich_data')

        parser.set_defaults(no_dropout=True)
        # To avoid cropping, the loadSize should be the same as fineSize
        parser.set_defaults(loadSize=parser.get_default('fineSize'))
        parser.set_defaults(dataroot='demo')

        
        self.isTrain = False
        return parser
