from PIL import Image
import joint_transforms
import torch.utils.data as data
from torchvision import transforms


joint_transform = joint_transforms.Compose([
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.RandomRotate(90),
    joint_transforms.Resize((256, 256))
])

val_joint_transform = joint_transforms.Compose([
    joint_transforms.Resize((256, 256)),
])


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

label_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()


class ImageFolder(data.Dataset):
    def __init__(self, image_list, mode='train', joint_transform=None, img_transform=None, label_transform=None):
        self.imgs = image_list
        self.joint_transform = joint_transform
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.mode = mode

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path)

        if self.mode == 'train':
            img, target = self.joint_transform(img, target)  # for training, both im and label are resized
        else:
            img, target = self.joint_transform(img, target)  # for testing, both im and label are resized, but no other trans
        img = self.img_transform(img)  # to tensor and normalized
        target = self.label_transform(target)

        return img, target, img_path

    def __len__(self):
        return len(self.imgs)
