
class DINOTransform:

    def __init__(
        self,
        global_crop_size: int = 224,
        global_crop_scale: tuple[float, float] = (0.4, 1.0),
        local_crop_size: int = 96,
        local_crop_scale: tuple[float, float] = (0.05, 0.4),
        n_local_views: int = 6,
    ):
        pass


class DINOViewTransform:
    def __init__(
        self,
        crop_size: int = 224,
        crop_scale: tuple[float, float] = (0.4, 1.0),
    ):
        transform = [
            T.RandomResizedCrop(
                size=crop_size,
                scale=crop_scale,
                # Type ignore needed because BICUBIC is not recognized as an attribute.
                interpolation=PIL.Image.BICUBIC,  # type: ignore[attr-defined]
            ),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            random_rotation_transform(rr_prob=rr_prob, rr_degrees=rr_degrees),
            T.RandomApply(
                [
                    T.ColorJitter(
                        brightness=cj_strength * cj_bright,
                        contrast=cj_strength * cj_contrast,
                        saturation=cj_strength * cj_sat,
                        hue=cj_strength * cj_hue,
                    )
                ],
                p=cj_prob,
            ),
            T.RandomGrayscale(p=random_gray_scale),
            GaussianBlur(
                kernel_size=kernel_size,
                scale=kernel_scale,
                sigmas=sigmas,
                prob=gaussian_blur,
            ),
            RandomSolarization(prob=solarization_prob),
            T.ToTensor(),
        ]

        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        self.transform = T.Compose(transform)

    def __call__(self, image: Union[Tensor, Image]) -> Tensor:
        """
        Applies the transforms to the input image.

        Args:
            image:
                The input image to apply the transforms to.

        Returns:
            The transformed image.

        """
        transformed: Tensor = self.transform(image)
        return transformed