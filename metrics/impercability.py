import numpy as np
from skimage.metrics import structural_similarity as ssim

class SteganographyMetrics:
    """
    A class to calculate and hold the quality metrics between an original
    image and a steganography-modified (stego) image.

    This version is optimized for RGB (multichannel) images.
    """
    def __init__(self, original_image: np.ndarray, stego_image: np.ndarray):
        """
        Initializes the metrics calculator with the two images to compare.

        Args:
            original_image (np.ndarray): The original image before modification.
            stego_image (np.ndarray): The image after data has been embedded.
        """
        # Ensure images are float type for accurate calculations
        self.original_image = original_image.astype(np.float64)
        self.stego_image = stego_image.astype(np.float64)

    def calculate_mse(self) -> float:
        """
        Calculates the Mean Squared Error (MSE) between the two images.
        MSE measures the average of the squares of the errors.
        A lower MSE value means less error and higher similarity.
        """
        return np.mean((self.original_image - self.stego_image) ** 2)

    def calculate_psnr(self, mse: float = None) -> float:
        """
        Calculates the Peak Signal-to-Noise Ratio (PSNR) from MSE.
        PSNR is used to measure the quality of reconstruction.
        A higher PSNR value (in dB) generally indicates better quality.

        Args:
            mse (float, optional): An already-calculated MSE.
                                     If None, MSE will be calculated.
        """
        if mse is None:
            mse = self.calculate_mse()

        if mse == 0:
            # The images are identical, PSNR is infinite.
            return float('inf')

        max_pixel_value = 255.0
        psnr_value = 20 * np.log10(max_pixel_value / np.sqrt(mse))
        return psnr_value

    def calculate_ssim(self) -> float:
        """
        Calculates the Structural Similarity Index (SSIM) between the two RGB images.
        SSIM measures the similarity in terms of structure, luminance, and contrast.
        The value ranges from -1 to 1, where 1 means identical images.
        """
        # data_range must be specified for float images
        data_range = 255.0

        # Since we know images are RGB, we can call ssim directly
        # with multichannel=True and specify the channel_axis.
        return ssim(self.original_image, self.stego_image,
                    multichannel=True,
                    data_range=data_range,
                    channel_axis=2) # Assumes shape (H, W, C)

    def get_all_metrics(self) -> dict:
        """
        A convenience method to calculate all metrics at once.

        Returns:
            dict: A dictionary containing the MSE, PSNR, and SSIM values.
        """
        # Calculate MSE once
        mse = self.calculate_mse()

        # Pass the calculated MSE to psnr
        psnr = self.calculate_psnr(mse)

        # Calculate SSIM
        ssim_val = self.calculate_ssim()

        print(f"MSE: {mse}")
        print(f"PSNR: {psnr} dB")
        print(f"SSIM: {ssim_val}")

        return {
            'mse': mse,
            'psnr': psnr,
            'ssim': ssim_val
        }