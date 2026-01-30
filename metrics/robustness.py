import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from helpers.message_binary import message_to_binary

class RobustnessTester:
    def __init__(self, steganography_instance, original_binary_message: str):
        self.steganography_instance = steganography_instance
        self.original_binary_message = original_binary_message
        self.bit_length = len(original_binary_message)

        # Maps the 'name' from configurations to a specific attack function
        self.attack_map = {
            'JPEG': self._apply_jpeg_compression,
            'Gaussian_Noise': self._apply_gaussian_noise,
            'SP_Noise': self._apply_salt_pepper_noise,
            'Crop': self._apply_crop,
            'Rotate': self._apply_rotation,
            'Scale': self._apply_downscale_upscale,
            'Blur': self._apply_gaussian_blur,
            'Sharpen': self._apply_sharpening,
            'HistEq': self._apply_histogram_equalization,
        }

    def _calculate_ber(self, extracted_message: str) -> float:
        """Calculates the Bit Error Rate between original and extracted messages."""
        if not extracted_message:
            return 1.0

        try:
             extracted_binary = message_to_binary(extracted_message)
             if not extracted_binary:
                return 1.0
        except Exception as e:
             print(f"Error saat konversi biner: {e}. Mengasumsikan BER 1.0")
             return 1.0


        original_binary = self.original_binary_message[:self.bit_length]
        extracted_binary = extracted_binary[:self.bit_length]

        if len(extracted_binary) < len(original_binary):
            extracted_binary = extracted_binary.ljust(len(original_binary), '0')

        errors = sum(1 for a, b in zip(original_binary, extracted_binary) if a != b)
        return errors / self.bit_length if self.bit_length > 0 else 0.0

    def run_all_tests(self,
                        stego_image: np.ndarray,
                        attack_configurations: list,
                        bit_lengths=None) -> dict:
        """
        Menjalankan semua uji ketahanan.
        Asumsi stego_image adalah np.ndarray dalam format RGB.

        Args:
            stego_image (np.ndarray): Gambar yang berisi pesan tersembunyi.
            attack_configurations (list): Daftar konfigurasi serangan.
            bit_lengths (any, optional): Argumen panjang bit untuk metode extract.
            
        Returns:
            dict: Kamus yang memetakan label serangan ke (BER, attacked_image_array).
                  Contoh: {'JPEG_90': (0.01, <np.ndarray>), 'Crop_10': (0.5, <np.ndarray>)}
                  # <--- PERUBAHAN DOCSTRING DI SINI ---
        """
        robustness_results = {}
        print(f"--- Menjalankan Uji Ketahanan ---")

        extraction_arg = bit_lengths if bit_lengths is not None else self.bit_length

        for config in attack_configurations:
            attack_name = config.get('name')
            attack_label = config.get('label', attack_name)
            attack_function = self.attack_map.get(attack_name)
            if not attack_function:
                print(f"  - ⚠️ Peringatan: Serangan '{attack_name}' tidak ditemukan. Dilewati.")
                continue

            params = {k: v for k, v in config.items() if k not in ['name', 'label']}
            try:
                # Fungsi serangan menerima RGB dan mengembalikan RGB
                attacked_image = attack_function(stego_image.copy(), **params)

                # Ekstraksi dijalankan pada gambar RGB yang diserang
                extracted_message = self.steganography_instance.extract(attacked_image, extraction_arg)

                ber = self._calculate_ber(extracted_message)
                
                # --- PERUBAHAN DI SINI: Simpan BER dan gambar ---
                robustness_results[attack_label] = (ber, attacked_image)
                # --- AKHIR PERUBAHAN ---
                
                print(f"  - ✅ Selesai: {attack_label:<20} | BER: {ber:.4f}")
            except Exception as e:
                print(f"  - ❌ Error saat serangan '{attack_label}': {e}")
                
                # --- PERUBAHAN DI SINI: Kembalikan None untuk gambar jika gagal ---
                robustness_results[attack_label] = (1.0, None)
                # --- AKHIR PERUBAHAN ---

        print("--- Semua pengujian selesai. ---")
        return robustness_results

    # --- Static Attack Methods (Input/Output adalah RGB) ---
    # (Metode-metode ini tidak berubah)
    @staticmethod
    def _apply_jpeg_compression(image, quality=95):
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, enc_img = cv2.imencode('.jpg', image_bgr, encode_param)
        dec_img_bgr = cv2.imdecode(enc_img, cv2.IMREAD_COLOR)
        return cv2.cvtColor(dec_img_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _apply_gaussian_noise(image, sigma=25):
        gauss = np.random.normal(0, sigma, image.shape)
        noisy = image.astype(np.float32) + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_salt_pepper_noise(image, density=0.01):
        output = image.copy()
        h, w, c = output.shape
        if c != 3:
             print("Peringatan SP_Noise: Gambar bukan 3-channel. Melewatkan.")
             return image
        num_pixels = h * w
        num_salt = int(np.ceil(density * num_pixels * 0.5))
        salt_coords_h = np.random.randint(0, h - 1, num_salt)
        salt_coords_w = np.random.randint(0, w - 1, num_salt)
        output[salt_coords_h, salt_coords_w] = (255, 255, 255)
        num_pepper = int(np.ceil(density * num_pixels * 0.5))
        pepper_coords_h = np.random.randint(0, h - 1, num_pepper)
        pepper_coords_w = np.random.randint(0, w - 1, num_pepper)
        output[pepper_coords_h, pepper_coords_w] = (0, 0, 0)
        return output

    @staticmethod
    def _apply_crop(image, crop_percent=10):
        h, w = image.shape[:2]
        start_x = int(w * crop_percent / 200)
        start_y = int(h * crop_percent / 200)
        end_x = w - start_x
        end_y = h - start_y
        cropped = image[start_y:end_y, start_x:end_x]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _apply_rotation(image, angle=5):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    @staticmethod
    def _apply_downscale_upscale(image, downscale_factor=0.7):
        h, w = image.shape[:2]
        new_dims = (int(w * downscale_factor), int(h * downscale_factor))
        downscaled = cv2.resize(image, new_dims, interpolation=cv2.INTER_AREA)
        return cv2.resize(downscaled, (w, h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _apply_gaussian_blur(image, kernel_size=(5, 5)):
        k_w = kernel_size[0] if kernel_size[0] % 2 != 0 else kernel_size[0] + 1
        k_h = kernel_size[1] if kernel_size[1] % 2 != 0 else kernel_size[1] + 1
        return cv2.GaussianBlur(image, (k_w, k_h), 0)

    @staticmethod
    def _apply_sharpening(image):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)

    @staticmethod
    def _apply_histogram_equalization(image):
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    
    
    
# --- Tentukan Konfigurasi Serangan yang Lengkap ---

ATTACK_CONFIGURATIONS = [
    # 1. Serangan Kompresi
    {'name': 'JPEG', 'label': 'JPEG (Q=90)', 'quality': 90},
    {'name': 'JPEG', 'label': 'JPEG (Q=70)', 'quality': 70},
    {'name': 'JPEG', 'label': 'JPEG (Q=50)', 'quality': 50},
    {'name': 'JPEG', 'label': 'JPEG (Q=30)', 'quality': 30},
    {'name': 'JPEG', 'label': 'JPEG (Q=10)', 'quality': 10},

    # 2. Serangan Noise
    {'name': 'Gaussian_Noise', 'label': 'Gaussian (σ=10)', 'sigma': 10},
    {'name': 'Gaussian_Noise', 'label': 'Gaussian (σ=25)', 'sigma': 25},

    {'name': 'SP_Noise', 'label': 'S&P (d=0.01)', 'density': 0.01},
    {'name': 'SP_Noise', 'label': 'S&P (d=0.03)', 'density': 0.03},

    # 3. Serangan Geometris
    {'name': 'Crop', 'label': 'Crop (3%)', 'crop_percent': 3},
    {'name': 'Crop', 'label': 'Crop (5%)', 'crop_percent': 5},
    {'name': 'Crop', 'label': 'Crop (10%)', 'crop_percent': 10},

    {'name': 'Rotate', 'label': 'Rotate (3°)', 'angle': 3},
    {'name': 'Rotate', 'label': 'Rotate (5°)', 'angle': 5},
    {'name': 'Rotate', 'label': 'Rotate (9°)', 'angle': 9},
    {'name': 'Rotate', 'label': 'Rotate (18°)', 'angle': 18},
    {'name': 'Rotate', 'label': 'Rotate (36°)', 'angle': 36},

    {'name': 'Scale', 'label': 'Scale (5%)', 'downscale_factor': 0.05},
    {'name': 'Scale', 'label': 'Scale (10%)', 'downscale_factor': 0.1},
    {'name': 'Scale', 'label': 'Scale (20%)', 'downscale_factor': 0.2},
    {'name': 'Scale', 'label': 'Scale (40%)', 'downscale_factor': 0.4},
    {'name': 'Scale', 'label': 'Scale (50%)', 'downscale_factor': 0.50},
    {'name': 'Scale', 'label': 'Scale (75%)', 'downscale_factor': 0.75},
    {'name': 'Scale', 'label': 'Scale (60%)', 'downscale_factor': 0.6},
    {'name': 'Scale', 'label': 'Scale (80%)', 'downscale_factor': 0.8},
    {'name': 'Scale', 'label': 'Scale (100%)', 'downscale_factor': 1.0},
    {'name': 'Scale', 'label': 'Scale (120%)', 'downscale_factor': 1.2},
    {'name': 'Scale', 'label': 'Scale (150%)', 'downscale_factor': 1.5},
    {'name': 'Scale', 'label': 'Scale (200%)', 'downscale_factor': 2.0},

    # 4. Serangan Filter & Warna
    {'name': 'Blur', 'label': 'Blur (3x3)', 'kernel_size': (3, 3)},
    {'name': 'Blur', 'label': 'Blur (5x5)', 'kernel_size': (5, 5)},
    {'name': 'Blur', 'label': 'Blur (9x9)', 'kernel_size': (9, 9)},
    {'name': 'Blur', 'label': 'Blur (15x15)', 'kernel_size': (15, 15)},
    {'name': 'Blur', 'label': 'Blur (21x21)', 'kernel_size': (21, 21)},
    {'name': 'Blur', 'label': 'Blur (27x27)', 'kernel_size': (27, 27)},
    {'name': 'Blur', 'label': 'Blur (33x33)', 'kernel_size': (33, 33)},
    {'name': 'Blur', 'label': 'Blur (39x39)', 'kernel_size': (39, 39)},

    {'name': 'Sharpen', 'label': 'Sharpen'},
    {'name': 'HistEq', 'label': 'Hist. Equalize'},
]

def calculate_ber(original_text: str, extracted_text: str) -> float:
    """Calculates the Bit Error Rate between two text strings."""
    if not extracted_text:
        return 1.0  # Maximum error if extraction fails

    original_binary = message_to_binary(original_text)
    extracted_binary = message_to_binary(extracted_text)

    total_bits = len(original_binary)
    if total_bits == 0:
        return 0.0

    errors = 0
    compare_len = min(len(original_binary), len(extracted_binary))
    for i in range(compare_len):
        if original_binary[i] != extracted_binary[i]:
            errors += 1

    # Add errors for any difference in length
    errors += abs(len(original_binary) - len(extracted_binary))

    return errors / total_bits
