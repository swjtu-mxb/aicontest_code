import cv2
import numpy as np
import sys
 
import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
 
import src.host.change_detec.vitis_model as cdp
 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def predict(argv):
    model = cdp.Unet1(
        encoder_name="resnet18",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=2,  # model output channels (number of classes in your datasets)
        siam_encoder=True,  # whether to use a siamese encoder
        fusion_form='concat',  # the form of fusing features from two branches. e.g. concat, sum, diff, or abs_diff.
    )
    
    model_path = './datas/ckpt/best_model.pth'
    model = torch.load(model_path)
    model.to(DEVICE)
    model.eval()
    
    test_transform = A.Compose([
                A.Normalize()])
    
    path1 = './datas/images/A/test_' + str(argv[1]) + '.png'
    img1 = cv2.imread(path1)
    img1 = test_transform(image = img1)['image'].transpose(2, 0, 1)
    img1 = torch.Tensor(np.expand_dims(img1,0)).to(DEVICE)
  
    
    path2 = './datas/images/B/test_' + str(argv[1]) + '.png'
    img2 = cv2.imread(path2)
    img2 = test_transform(image = img2)['image'].transpose(2, 0, 1)
    img2 = torch.Tensor(np.expand_dims(img2,0)).to(DEVICE)
    
    
    pre = model(img1,img2)
    pre = torch.argmax(pre, dim=1).cpu().data.numpy()
    pre = pre * 255.0
    path3 = './datas/images/result/test_' + str(argv[1]) + '_pre.png'
    cv2.imwrite(path3, pre[0])
    # img1 = cv2.imread(path3)
    # cv2.imshow('result of change detection', img1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    predict(sys.argv)
    
