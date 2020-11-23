import albumentations


def get_transforms(image_size):

    transforms_train = albumentations.Compose([

        albumentations.HorizontalFlip(p=0.5),
        # albumentations.RandomBrightnessContrast(p=0.75),
        # albumentations.RandomContrast(limit=0.2, p=0.75),
        albumentations.OneOf([
            albumentations.MedianBlur(blur_limit=5),
            albumentations.GaussianBlur(blur_limit=5),
            albumentations.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),

        # albumentations.OneOf([
        #     albumentations.OpticalDistortion(distort_limit=1.0),
        #     albumentations.GridDistortion(num_steps=5, distort_limit=1.),
        #     albumentations.ElasticTransform(alpha=3),
        # ], p=0.7),

        albumentations.CLAHE(clip_limit=4.0, p=0.3),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.3),
        albumentations.Resize(image_size, image_size),
        albumentations.CoarseDropout(p=0.3, max_holes=1, max_height=75, max_width=75),
        albumentations.Normalize(mean=(0.5964188, 0.4566936, 0.3908954), std=(0.2590655, 0.2314241, 0.2269535))
    ])

    transforms_val = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize(mean=(0.5964188, 0.4566936, 0.3908954), std=(0.2590655, 0.2314241, 0.2269535))
    ])

    return transforms_train, transforms_val