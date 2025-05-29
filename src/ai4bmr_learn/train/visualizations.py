# if self.current_epoch % 50 == 0 and batch_idx == 0:
#     print(
#         f"Logging samples at Epoch {self.current_epoch} and batch {batch_idx}"
#     )
#     num_samples = 4
#     num_channels = images.shape[1]
#
#     images = images[:num_samples].detach().cpu().float()
#     predictions = predictions[:num_samples].detach().cpu().float()
#     masks = masks[:num_samples].detach().cpu()
#
#     channel_names = batch.get("channel_names", None)
#     if channel_names is not None:
#         channel_names = [i[0] for i in channel_names]
#     else:
#         channel_names = [f"channel_{i}" for i in range(num_channels)]
#
#     self.log_composition(
#         images=images,
#         predictions=predictions,
#         masks=masks,
#         channel_names=channel_names,
#     )
#     if num_channels == 3:
#         self.log_composition_3D(
#             images=images,
#             predictions=predictions,
#             masks=masks,
#             channel_names=channel_names,
#         )


    def log_reconstructions(
            self, *, images, predictions, masks, channel_names: list[str] = None
    ):
        from torchvision.utils import make_grid

        # full reconstruction
        reconstruction = predictions.clone()
        reconstruction[~masks.bool()] = images[~masks.bool()]

        grid = make_grid(reconstruction, nrow=3)
        self.logger.log_image(
            key=f"reconstruction-v1/",
            images=[grid],
            caption=["Reconstruction"],
        )

    def log_composition(self, images, predictions, channel_names, masks=None):

        for i, (img, pred, mask) in enumerate(zip(images, predictions, masks)):
            img_cols, caption = grid_composition(
                img, pred, mask, channel_names=channel_names
            )

            C, height, width = img.shape
            img = img[mask.bool()].reshape(C, -1)
            pred = pred[mask.bool()].reshape(C, -1)

            hist_col = image_to_dist_grid(
                image=img,
                prediction=pred,
                channel_names=channel_names,
                kind="hist",
                height=height,
                width=width,
                nrow=1,
                return_list=False,
            )

            grid = torch.cat([img_cols, hist_col], dim=-1)
            caption += " | Histogram"

            self.logger.log_image(
                key=f"composition-v1/{i}",
                images=[grid],
                caption=[caption],
            )

    def log_multi_channel_grid(self, images, predictions, channel_names, masks=None):

        for i, (img, pred, mask) in enumerate(zip(images, predictions, masks)):
            imgs, caps = grid_composition(img, pred, mask, channel_names=channel_names)

            self.logger.log_image(
                key=f"reconstructions-v1/{i}",
                images=[imgs],
                caption=[caps],
            )

    def log_channel_histograms(
            self, *, images, predictions, masks, channel_names: list[str]
    ):
        from visualization.multi_channel_image import image_to_dist_grid

        for i, (img, pred, mask) in enumerate(zip(images, predictions, masks)):
            # NOTE: we only keep the masked pixels
            C, height, width = img.shape
            img = img[mask.bool()].reshape(C, -1)
            pred = pred[mask.bool()].reshape(C, -1)

            hist_list = image_to_dist_grid(
                image=img,
                prediction=pred,
                channel_names=channel_names,
                kind="hist",
                height=height,
                width=width,
                return_list=True,
            )

            self.logger.log_image(
                key=f"histograms/",
                images=hist_list,
                caption=channel_names,
            )


from visualization.multi_channel_image import image_to_channels_grid, image_to_dist_grid


def grid_composition(img, pred, mask, channel_names: list[str] = None):
    g0, _, _ = image_to_channels_grid(
        img,
        nrow=1,
        normalize=True,
        scale_each=True,
        cmap_name="inferno",
        channel_names=channel_names,
    )

    if mask is not None:
        pred[~mask.bool()] = 0
        g1, _, _ = image_to_channels_grid(
            pred,
            nrow=1,
            normalize=True,
            scale_each=True,
            cmap_name="inferno",
            channel_names=channel_names,
        )

        img_masked = img.clone()
        img_masked[mask.bool()] = 0

        g2, _, _ = image_to_channels_grid(
            img_masked,
            nrow=1,
            normalize=True,
            scale_each=True,
            cmap_name="inferno",
            channel_names=channel_names,
        )

        reconstruction = pred.clone()
        reconstruction[~mask.bool()] = img[~mask.bool()]

        g3, _, _ = image_to_channels_grid(
            reconstruction,
            nrow=1,
            normalize=True,
            scale_each=True,
            cmap_name="inferno",
            channel_names=channel_names,
        )

        g = torch.cat([g0, g3, g1, g2], dim=2)
        caption = "Image | Reconstruction | Prediction | Masked Prediction"
    else:
        g1, _, _ = image_to_channels_grid(
            pred,
            nrow=1,
            normalize=True,
            scale_each=True,
            cmap_name="inferno",
            channel_names=channel_names,
        )

        g = torch.cat([g0, g1], dim=2)
        caption = "Image | Reconstruction"

    return g, caption
