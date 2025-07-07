import os
from skimage import io
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from u2net.u2net_test import normPRED
from u2net.data_loader import RescaleT, ToTensorLab, SalObjDataset
import numpy as np
from PIL import Image
import glob
import warnings

warnings.filterwarnings("ignore")

def save_images(image_name, pred, d_dir):
    predict = pred.squeeze().cpu().data.numpy()
    im = Image.fromarray(predict * 255).convert('RGB')
    img_name = os.path.basename(image_name)
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BICUBIC)
    imidx = ".".join(img_name.split(".")[:-1])
    print('Saving output at {}'.format(os.path.join(d_dir, imidx + '.png')))
    imo.save(os.path.join(d_dir, imidx + '.png'))

def infer(
    net,
    image_dir=os.path.join(os.getcwd(), 'test_data', 'test_images'),
    prediction_dir=os.path.join(os.getcwd(), 'test_data', 'u2net_results')
):
    img_name_list = glob.glob(image_dir + os.sep + '*')
    prediction_dir = prediction_dir + os.sep

    test_salobj_dataset = SalObjDataset(
        img_name_list=img_name_list,
        lbl_name_list=[],
        transform=transforms.Compose([
            RescaleT(320),
            ToTensorLab(flag=0)
        ])
    )

    test_salobj_dataloader = DataLoader(
        test_salobj_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0  # ✅ Fix for Windows multiprocessing
    )

    for i_test, data_test in enumerate(test_salobj_dataloader):
        print("Generating mask for:", os.path.basename(img_name_list[i_test]))

        inputs_test = data_test['image'].type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)
        pred = normPRED(d1[:, 0, :, :])

        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)

        save_images(img_name_list[i_test], pred, prediction_dir)

        del d1, d2, d3, d4, d5, d6, d7
