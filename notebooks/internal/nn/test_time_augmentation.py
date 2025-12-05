import torchvision.transforms.functional as TF

def apply_tta(img_tensor):
    """
    img_tensor: [3, H, W] torch tensor (after ToTensor+Normalize)
    returns list of augmented tensors (each [3,H,W])
    """
    tta_list = []

    # identity
    tta_list.append(img_tensor)

    # horizontal flip
    tta_list.append(TF.hflip(img_tensor))

    # vertical flip
    tta_list.append(TF.vflip(img_tensor))

    # rotations
    tta_list.append(TF.rotate(img_tensor, 90))
    tta_list.append(TF.rotate(img_tensor, -90))
    tta_list.append(TF.rotate(img_tensor, 180))

    return tta_list
