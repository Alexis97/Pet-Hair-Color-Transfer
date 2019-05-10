import os
from options.demo_options import DemoOptions
from data import CreateDataLoader
from models import create_model
from data.base_dataset import get_transform
from util.util import tensor2im 
from PIL import Image
import glob
import time

IMG_DIR = './datasets/demo/dog_orange_long_hair'
MASK_DIR = './datasets/demo/dog_orange_long_hair_mask'
DEMO_DIR = './demo'
MAX_DEMO_NUM = 50

def createModel(opt):
    model = create_model(opt)
    model.setup(opt)
    return model


def dataPreprocess(opt, img_path, mask_path):
    img = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')

    r, g, b = img.split()
    img_cat = Image.merge("RGBA", [r,g,b,mask])

    transform = get_transform(opt)
    img_cat = transform(img_cat)

    img = img_cat[0:3, :, :].unsqueeze(0)
    mask = img_cat[3, :, :].unsqueeze(0).unsqueeze(0)

    return {'img': img, 'mask': mask}



if __name__ == '__main__':
    opt = DemoOptions().parse()
    opt.which_direction = 'BtoA'
    model = createModel(opt)

    if not os.path.exists(DEMO_DIR):
        os.makedirs(DEMO_DIR)

    count = 0
    for img_path in glob.glob(os.path.join(IMG_DIR, '*')):
        if (count >= MAX_DEMO_NUM):
            break
        time_1 = time.time()
        fname = os.path.split(img_path)[-1].split('.')[0]

        mask_path = os.path.join(MASK_DIR, fname+'.png')
        result_path = os.path.join(DEMO_DIR, fname+'_fake.jpg')

        data = dataPreprocess(opt, img_path, mask_path)

        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        demo_result = visuals['fake']
        demo_result = tensor2im(demo_result)
        demo_result = Image.fromarray(demo_result)
        demo_result.save(result_path)
        print ('Cost %.3f secs, save demo to:%s' % (time.time() - time_1, result_path))

        count += 1

