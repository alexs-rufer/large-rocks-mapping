import json
from torch.utils.data import Dataset
from tifffile import imread
import numpy as np
import utils.paths as paths
from sklearn.decomposition import PCA

"""
Dataset and transform classes for supervised training.

- RockDetectionDataset loads image-annotation samples from a JSON file.
- Several transformation classes are provided to fuse RGB and hillshade inputs,
  including channel replacement, PCA fusion, multiplication, etc.

These are used before preparing YOLO-compatible training files.
"""

class RockDetectionDataset(Dataset):
    """
    Custom Dataset for rock detection.
    
    Args:
        json_file (str): Path to the JSON file with annotations.
        root_dir (str): Directory where the images are stored.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    
    def __init__(self, json_file, root_dir, transform=None):
        # Load data set from json file
        with open(json_file, 'r') as f:
            data = json.load(f)

        self.samples = data["dataset"]
        self.info = data["info"]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # get sample 
        sample = self.samples[idx]

        # default image value is RGB if no transform is applied
        file_name = sample["file_name"]
        path_SI = self.root_dir / "swissImage_50cm_patches" / file_name
        image_SI = imread(path_SI)
        sample["image"] = image_SI
        
        # apply transform if any provided
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    

# TRANSFORMS - used to modify the dataset samples before training, best meethod is to replace the green Channel (channel 1) with the hillshade value.

# transform to combine RGB and HS images with a given alpha
class CombineRGBHillshade:
    def __init__(self, alpha=1.0):
        """
        Args:
            alpha (float): Weight of the RGB image. (1 - alpha) is used for hillshade.
        """
        self.alpha = alpha

    def __call__(self, sample):
        # extract RGB and hillshade images from the sample.
        RGB_path = paths.RAW_DATA_DIR / 'swissImage_50cm_patches' / sample["file_name"]
        HS_path = paths.RAW_DATA_DIR / 'swissSURFACE3D_hillshade_patches' / sample["file_name"]

        RGB_image = imread(RGB_path)
        HS_image = imread(HS_path)

        # transform hillshade in 3 channel and make sure its uint8 like RGB
        HS_image = np.stack([HS_image, HS_image, HS_image], axis=-1).astype(RGB_image.dtype)

        # combine images and again make sure its uint8
        combined_image = (self.alpha * RGB_image + (1 - self.alpha) * HS_image).astype(RGB_image.dtype)

        # update sample accordingly
        sample["image"] = combined_image
        return sample

# Transform to replace one of the RGB channels with the HS value
class ReplaceRGBChannelWithHS:
    def __init__(self, channel=-1):
        """
        Args:
            channel (int): The index of the RGB channel to replace 
                           (-1 for no transform, 0 for red, 1 for green, or 2 for blue).
        """
        self.channel = channel

    def __call__(self, sample):
        if(self.channel == -1): return sample

        # Define paths to the RGB and hillshade images using the sample's file name.
        HS_path = paths.RAW_DATA_DIR / 'swissSURFACE3D_hillshade_patches' / sample["file_name"]

        # Read the images.
        RGB_image = sample["image"]
        HS_image = imread(HS_path)  # This is assumed to be a single-channel image

        # Ensure the hillshade image is the same type as the RGB image.
        HS_image = HS_image.astype(RGB_image.dtype)

        # Replace the specified channel in the RGB image with the hillshade image.
        RGB_image[..., self.channel] = HS_image

        # Update the sample with the modified image.
        sample["image"] = RGB_image
        return sample
    

# Transform to set RGB pixels to black where the hillshade value exceeds a threshold
class NonLinearHSBlackout:
    def __init__(self, threshold=150):
        """
        Args:
            threshold (int): Hillshade pixel value above which the corresponding RGB pixel will be set to black.
        """
        self.threshold = threshold

    def __call__(self, sample):
        # Define paths for the RGB and hillshade images using the sample's file name.
        HS_path = paths.RAW_DATA_DIR / 'swissSURFACE3D_hillshade_patches' / sample["file_name"]

        # Read the images.
        img = sample['image']
        img_hill = imread(HS_path)

        # Ensure the hillshade image is in uint8 format.
        img_hill = img_hill.astype(np.uint8)

        # Create a copy of the RGB image for the final output.
        final_image = img.copy()

        # Create a boolean mask where the hillshade is above the threshold.
        mask = img_hill < self.threshold

        # Set corresponding pixels in the final image to black.
        final_image[mask] = [0, 0, 0]

        # Update the sample with the modified image.
        sample["image"] = final_image
        return sample
    

class MultiplyRGBHS:
    def __init__(self):
        """
        Transform that scales both the RGB and HS images to [0,1],
        multiplies them elementwise, and scales the result back to [0,255].
        """
        pass

    def __call__(self, sample):
        # Define paths for the RGB and hillshade images
        RGB_path = paths.RAW_DATA_DIR / 'swissImage_50cm_patches' / sample["file_name"]
        HS_path = paths.RAW_DATA_DIR / 'swissSURFACE3D_hillshade_patches' / sample["file_name"]

        # Read the images.
        RGB_image = imread(RGB_path)
        HS_image = imread(HS_path)

        # If HS image is single-channel, replicate to match RGB shape.
        if HS_image.ndim == 2:
            HS_image = np.stack([HS_image, HS_image, HS_image], axis=-1)

        # Convert images to float32 and normalize to [0, 1]
        rgb_norm = RGB_image.astype(np.float32) / 255.0
        hs_norm = HS_image.astype(np.float32) / 255.0

        # Multiply both normalized images elementwise
        combined_norm = rgb_norm * hs_norm

        # Scale the result back to [0, 255] and convert to uint8
        combined_image = np.clip(combined_norm * 255, 0, 255).astype(np.uint8)

        # Update the sample with the combined image.
        sample["image"] = combined_image
        return sample
    
class PCAFusion:
    def __init__(self):
        """
        Fusion transform that combines an RGB image with a hillshade image using PCA.
        The RGB image (H, W, 3) is combined with the hillshade image (H, W) by stacking
        the hillshade (replicated to one channel) to create a 4-channel input for PCA.
        """
        pass

    def __call__(self, sample):
        # Construct file paths using the sample's file name.
        RGB_path = paths.RAW_DATA_DIR / 'swissImage_50cm_patches' / sample["file_name"]
        HS_path = paths.RAW_DATA_DIR / 'swissSURFACE3D_hillshade_patches' / sample["file_name"]

        # Read the images.
        RGB_img = imread(RGB_path)   # Expected shape: (H, W, 3)
        HS_img = imread(HS_path)     # Expected shape: (H, W)

        # Ensure the hillshade image is the same data type as the RGB image.
        HS_img = HS_img.astype(RGB_img.dtype)
        # Reshape hillshade to add a channel dimension: shape becomes (H, W, 1)
        HS_img = HS_img[..., np.newaxis]

        # Stack the RGB image and hillshade image along the channel axis to form a 4-channel image.
        combined = np.concatenate([RGB_img, HS_img], axis=-1)  # shape: (H, W, 4)

        # Reshape the combined image so that each pixel is a sample with 4 features.
        H, W, C = combined.shape
        flat = combined.reshape(-1, C)  # shape: (H*W, 4)

        # Apply PCA to reduce from 4 channels to 3 channels.
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(flat)  # shape: (H*W, 3)

        # Reshape back to image dimensions.
        fused = pca_result.reshape(H, W, 3)

        # Normalize the fused image to the 0-255 range.
        fused = fused - fused.min()
        fused = fused / fused.max() * 255
        fused = fused.astype(np.uint8)

        # Update the sample with the new fused image.
        sample["image"] = fused
        return sample