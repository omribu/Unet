import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms


# root_path = /home/omri/Unet/data/plowing

class DataSets(Dataset):
    def __init__(self, root_path, test=False):
        self.root_path = root_path
        if test:
            self.images = sorted([root_path+"/test/"+i for i in os.listdir(root_path+"/test/")])
            self.masks = sorted([root_path+"/mask_test/"+i for i in os.listdir(root_path+"/mask_test/")])
        else:
            self.images = sorted([root_path+"/train/"+i for i in os.listdir(root_path+"/train/")])
            self.masks = sorted([root_path+"/mask_train/"+i for i in os.listdir(root_path+"/mask_train/")])

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()])
        
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")

        return self.transform(img), self.transform(mask)


    def __len__(self):
        return len(self.images)