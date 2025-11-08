import numpy as np
from helpers.message_binary import message_to_binary, binary_to_message

class PVDSteganography:
    """
    Implementasi PVD Steganography
    Versi ini DISEDERHANAKAN: hanya untuk gambar RGB (3-channel).
    """
    def __init__(self):
        self.ranges = [
            [0, 7, 3], [8, 15, 3], [16, 31, 4],
            [32, 63, 5], [64, 127, 6], [128, 255, 7]
        ]

    # _prepare_image method is REMOVED.

    def _calculate_capacity(self, image: np.ndarray) -> int:
        """
        Menghitung kapasitas embedding untuk gambar RGB.
        """
        # Asumsikan gambar adalah (H, W, 3)
        height, width, _ = image.shape
        total_capacity = 0
        for row in range(height):
            for col in range(0, width - 1, 2):
                # Loop eksplisit 3 channel (R, G, B)
                for channel in range(3):
                    p1, p2 = int(image[row, col, channel]), int(image[row, col + 1, channel])
                    diff = p2 - p1
                    _, capacity, _, _ = self.get_range_and_capacity(diff)
                    total_capacity += capacity
        return total_capacity

    def get_range_and_capacity(self, diff: int) -> tuple:
        """
        Mendapatkan rentang, kapasitas, dan batas untuk nilai perbedaan.
        """
        abs_diff = abs(diff)
        for i, (lower, upper, capacity) in enumerate(self.ranges):
            if lower <= abs_diff <= upper:
                return i, capacity, lower, upper
        return 0, 0, 0, 0 # Jika di luar rentang

    def embed(self, cover_image: np.ndarray, secret_message: str) -> tuple:
        """
        Menyisipkan pesan ke dalam gambar RGB.
        """
        # Asumsikan cover_image adalah (H, W, 3) RGB
        binary_message = message_to_binary(secret_message)
        message_length = len(binary_message)

        max_capacity = self._calculate_capacity(cover_image)
        if message_length > max_capacity:
            raise ValueError(f"Pesan terlalu panjang! Butuh {message_length} bits, kapasitas {max_capacity} bits.")

        height, width, _ = cover_image.shape
        stego_image = cover_image.copy().astype(np.float64)
        binary_index = 0

        for row in range(height):
            for col in range(0, width - 1, 2):
                if binary_index >= message_length: break
                # Loop eksplisit 3 channel (R, G, B)
                for channel in range(3):
                    if binary_index >= message_length: break

                    p1, p2 = int(stego_image[row, col, channel]), int(stego_image[row, col + 1, channel])
                    diff = p2 - p1
                    _, capacity, _, _ = self.get_range_and_capacity(diff)
                    if capacity == 0: continue

                    bits_needed = min(capacity, message_length - binary_index)
                    if bits_needed == 0: continue

                    message_segment = binary_message[binary_index : binary_index + bits_needed]
                    embed_value = int(message_segment, 2)

                    extracted_value = abs(diff) % (2**capacity)
                    remainder = abs(diff) - extracted_value
                    new_diff_val = remainder + embed_value
                    new_diff = new_diff_val if diff >= 0 else -new_diff_val

                    adjustment = new_diff - diff
                    adj1, adj2 = adjustment // 2, adjustment - (adjustment // 2)
                    p1_new, p2_new = p1 - adj1, p2 + adj2

                    # Penanganan Underflow/Overflow
                    if p1_new < 0:
                        p2_new += p1_new
                        p1_new = 0
                    elif p1_new > 255:
                        p2_new -= (p1_new - 255)
                        p1_new = 255

                    if p2_new < 0:
                        p1_new += p2_new
                        p2_new = 0
                    elif p2_new > 255:
                        p1_new -= (p2_new - 255)
                        p2_new = 255

                    # Failsafe clip
                    if p1_new < 0: p1_new = 0
                    if p1_new > 255: p1_new = 255
                    if p2_new < 0: p2_new = 0
                    if p2_new > 255: p2_new = 255

                    stego_image[row, col, channel] = p1_new
                    stego_image[row, col + 1, channel] = p2_new
                    binary_index += bits_needed

            final_stego_image = np.clip(stego_image, 0, 255).astype(np.uint8)
            return final_stego_image, message_length

    def extract(self, stego_image: np.ndarray, bit_length: int) -> str:
        """
        Mengekstrak pesan dari gambar stego RGB.
        """
        # Asumsikan stego_image adalah (H, W, 3) RGB
        height, width, _ = stego_image.shape
        bit_buffer = ""
        bits_extracted = 0

        for row in range(height):
            for col in range(0, width - 1, 2):
                if bits_extracted >= bit_length: break
                # Loop eksplisit 3 channel (R, G, B)
                for channel in range(3):
                    if bits_extracted >= bit_length: break

                    p1, p2 = int(stego_image[row, col, channel]), int(stego_image[row, col + 1, channel])
                    diff = p2 - p1
                    _, capacity, _, _ = self.get_range_and_capacity(diff)
                    if capacity == 0: continue

                    bits_to_take = min(capacity, bit_length - bits_extracted)
                    if bits_to_take <= 0:
                        # This can happen if bit_length is met exactly
                        break

                    extracted_value = abs(diff) % (2**capacity)

                    # --- PERBAIKAN BUG KRUSIAL ---
                    # Format nilai yang diekstrak ke 'bits_to_take' (bukan 'capacity')
                    # untuk mendapatkan jumlah bit yang benar dan padding nol yang benar.
                    extracted_bits = format(extracted_value, f'0{bits_to_take}b')
                    # ---------------------------

                    bit_buffer += extracted_bits
                    bits_extracted += bits_to_take

            # Kembalikan hanya bit yang diminta
            return binary_to_message(bit_buffer[:bit_length])
        
PVD_DEFAULT_PARAM = {}