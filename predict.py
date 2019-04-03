from argparse import ArgumentParser
from glob import iglob
from os.path import basename

from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
from segment_rl import DRNSeg,resize_4d_tensor,save_output_images


parser = ArgumentParser()
parser.add_argument('im-pattern')
parser.add_argument('arch')
parser.add_argument('-c', '--classes', default=0, type=int)
parser.add_argument('--pretrained', dest='pretrained',
                    default='', type=str, metavar='PATH',
                    help='use pre-trained model')
parser.add_argument('-o','--output-dir',default='results',metavar='PATH')

scales = [0.5, 0.75]
normalize = transforms.Normalize(
    std=[0.1829540508368939, 0.18656561047509476, 0.18447508988480435],
    mean=[0.29010095242892997, 0.32808144844279574, 0.28696394422942517])
transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

def preprocess(x):
    """
    x is a PIL image
    """
    w,h = x.size
    x = np.array(x)
    if len(x.shape) == 2:
        x = np.stack([x , x , x] , axis = 2)
    x = [Image.fromarray(x)]
    out_x = [transform(x)]
    ms_xs = [transform(x.resize((int(w * s), int(h * s)),Image.BICUBIC)) for s in scales]
    out_x.extend(ms_xs)
    return out_x

def loadim(fn):
    im = Image.open(fn)
    return im

class DRNSegPredict(DRNSeg):

    def forward(self,im):
        ims = preprocess(im)
        outputs = []
        with torch.no_grad():
            for im in ims:
                im_var = Variable(im,requires_grads=False)
                output = model(im_var)[0]
                outputs.append(output)
        final_output = sum([resize_4d_tensor(out, w, h) for out in outputs])
        pred = final_output.argmax(axis=1)
        return pred




def main():
    args = parser.parse_args()
    model =  DRNSegPredict(args.arch, args.classes, pretrained_model=None,
                          pretrained=False)
    model.load_state_dict(torch.load(args.pretrained))
    for fn in iglob(args.im_pattern):
        im = loadim(fn)
        pred = model(im)
        name = basename(fn)
        save_output_images(pred, name, args.output_dir)

if __name__ == '__main__':
    main()