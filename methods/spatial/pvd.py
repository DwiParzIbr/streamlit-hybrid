import numpy as np
# Asumsi helpers ada di folder yang sama atau disesuaikan import-nya
from helpers.message_binary import message_to_binary, binary_to_message

class PVDSteganography:
    """
    Implementasi PVD Steganography (FIXED Version)
    Perbaikan: Logika Boundary Check kini menggunakan pergeseran (shifting)
    yang menjaga nilai selisih (difference) tetap utuh.
    """
    def __init__(self):
        self.ranges = [
            [0, 7, 3], [8, 15, 3], [16, 31, 4],
            [32, 63, 5], [64, 127, 6], [128, 255, 7]
        ]

    def _calculate_capacity(self, image: np.ndarray) -> int:
        height, width, _ = image.shape
        total_capacity = 0
        for row in range(height):
            for col in range(0, width - 1, 2):
                for channel in range(3):
                    p1, p2 = int(image[row, col, channel]), int(image[row, col + 1, channel])
                    diff = p2 - p1
                    _, capacity, _, _ = self.get_range_and_capacity(diff)
                    total_capacity += capacity
        return total_capacity

    def get_range_and_capacity(self, diff: int) -> tuple:
        abs_diff = abs(diff)
        for i, (lower, upper, capacity) in enumerate(self.ranges):
            if lower <= abs_diff <= upper:
                return i, capacity, lower, upper
        return 0, 0, 0, 0

    def embed(self, cover_image: np.ndarray, secret_message: str) -> tuple:
        binary_message = message_to_binary(secret_message)
        message_length = len(binary_message)

        max_capacity = self._calculate_capacity(cover_image)
        if message_length > max_capacity:
            raise ValueError(f"Pesan terlalu panjang! Butuh {message_length} bits, kapasitas {max_capacity} bits.")

        height, width, _ = cover_image.shape
        # Gunakan int32 agar aman saat nilai negatif/lebih dari 255
        stego_image = cover_image.copy().astype(np.int32)
        binary_index = 0

        for row in range(height):
            for col in range(0, width - 1, 2):
                if binary_index >= message_length: break
                
                for channel in range(3):
                    if binary_index >= message_length: break

                    p1, p2 = stego_image[row, col, channel], stego_image[row, col + 1, channel]
                    diff = p2 - p1

                    _, capacity, _, _ = self.get_range_and_capacity(diff)
                    if capacity == 0: continue

                    bits_needed = min(capacity, message_length - binary_index)
                    if bits_needed == 0: continue

                    message_segment = binary_message[binary_index : binary_index + bits_needed]
                    embed_value = int(message_segment, 2)

                    # --- PROSES MATEMATIKA UTAMA ---
                    extracted_value = abs(diff) % (2**capacity)
                    remainder = abs(diff) - extracted_value
                    new_diff_abs = remainder + embed_value
                    
                    # Kembalikan tanda positif/negatif
                    new_diff = new_diff_abs if diff >= 0 else -new_diff_abs

                    adjustment = new_diff - diff
                    adj1 = adjustment // 2
                    adj2 = adjustment - adj1
                    
                    p1_new = p1 - adj1
                    p2_new = p2 + adj2

                    # --- PERBAIKAN BOUNDARY CHECK (CRITICAL FIX) ---
                    # Jangan di-clip paksa satu per satu.
                    # Geser KEDUANYA secara bersamaan agar selisih (diff) tidak berubah.

                    # 1. Cek Batas Bawah (< 0)
                    min_val = min(p1_new, p2_new)
                    if min_val < 0:
                        # Geser naik sebanyak nilai minusnya
                        offset = -min_val
                        p1_new += offset
                        p2_new += offset
                    
                    # 2. Cek Batas Atas (> 255)
                    max_val = max(p1_new, p2_new)
                    if max_val > 255:
                        # Geser turun sebanyak kelebihannya
                        offset = max_val - 255
                        p1_new -= offset
                        p2_new -= offset

                    # Simpan ke citra stego
                    stego_image[row, col, channel] = p1_new
                    stego_image[row, col + 1, channel] = p2_new
                    
                    binary_index += bits_needed

        # Kembalikan ke uint8 dengan aman karena boundary check sudah menjamin 0-255
        return stego_image.astype(np.uint8), message_length

    def extract(self, stego_image: np.ndarray, bit_length: int) -> str:
        height, width, _ = stego_image.shape
        bit_buffer = ""
        bits_extracted = 0
        
        # Casting ke int32 penting untuk perhitungan diff yang akurat
        stego_work = stego_image.astype(np.int32)

        for row in range(height):
            for col in range(0, width - 1, 2):
                if bits_extracted >= bit_length: break
                
                for channel in range(3):
                    if bits_extracted >= bit_length: break

                    p1, p2 = stego_work[row, col, channel], stego_work[row, col + 1, channel]
                    diff = p2 - p1

                    _, capacity, _, _ = self.get_range_and_capacity(diff)
                    if capacity == 0: continue

                    bits_to_take = min(capacity, bit_length - bits_extracted)
                    if bits_to_take <= 0: break

                    extracted_value = abs(diff) % (2**capacity)
                    
                    # Format biner sesuai jumlah bit yang diambil
                    extracted_bits = format(extracted_value, f'0{bits_to_take}b')
                    
                    bit_buffer += extracted_bits
                    bits_extracted += bits_to_take

        return binary_to_message(bit_buffer[:bit_length])

PVD_DEFAULT_PARAM = {}
