
from torch.utils.data import Dataset
from glob import glob
import cv2
import os

class CatsAndDogsDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.imagepath_list = glob( self.root + '/*.jpg')
    
    def __len__(self):
        return len(self.imagepath_list)
    
    def __getitem__(self, index):
        imagepath = self.imagepath_list[index]
        label = 0 if 'cat' in os.path.basename(imagepath) else 1 # Cat == 0 | Dog == 1

        image = cv2.imread(imagepath)[:,:,::-1] # read an image and convert from BGR to RGB format

        if self.transform is not None:
            image = self.transform(image)
        return image, label

def test():
    root = '/datasets/dogs-vs-cats/train'
    dataset = CatsAndDogsDataset(root)

    image,label = dataset[0]

    cv2.imwrite(f'{label}.jpg',image[:,:,::-1])

if __name__=='__main__':
    test()