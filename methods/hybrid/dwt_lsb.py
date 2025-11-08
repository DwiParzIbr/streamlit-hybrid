from methods.frequency.dwt import DWTSteganography
from methods.spatial.lsb import LSBSteganography
from helpers.message_binary import message_to_binary, binary_to_message

class DWTLSBHybrid:
    """
    Menggabungkan steganografi DWT dan LSB.
    """
    def __init__(self, dwt_lsb_ratio=(0.5, 0.5), dwt_params=None, lsb_params=None):
        """
        Inisialisasi metode hibrida.
        """
        if sum(dwt_lsb_ratio) != 1.0:
            raise ValueError("Jumlah dwt_lsb_ratio harus 1.0")

        self.dwt_ratio = dwt_lsb_ratio[0]
        self.lsb_ratio = dwt_lsb_ratio[1]

        self.dwt_steganography = DWTSteganography(**(dwt_params or {}))
        self.lsb_steganography = LSBSteganography(**(lsb_params or {}))

    def embed(self, cover_image, secret_message):
        """Menyisipkan pesan rahasia menggunakan metode hibrida DWT-LSB."""

        # 1. Bagi pesan
        full_binary_message = message_to_binary(secret_message)
        split_point = int(len(full_binary_message) * self.dwt_ratio)

        dwt_message_part = binary_to_message(full_binary_message[:split_point])
        lsb_message_part = binary_to_message(full_binary_message[split_point:])

        # 2. Sisipkan DWT (Input: RGB, Output: RGB)
        print(f"Hybrid: Menyisipkan {len(full_binary_message[:split_point])} bit via DWT...")
        intermediate_stego, dwt_bit_length = self.dwt_steganography.embed(
            cover_image.copy(), dwt_message_part
        )

        # 3. Sisipkan LSB (Input: RGB, Output: RGB)
        print(f"Hybrid: Menyisipkan {len(full_binary_message[split_point:])} bit via LSB...")
        final_stego, lsb_bit_length = self.lsb_steganography.embed(
            intermediate_stego, lsb_message_part
        )

        return final_stego, (dwt_bit_length, lsb_bit_length)

    def extract(self, stego_image, bit_lengths):
        """Mengekstrak pesan rahasia (LSB dulu, baru DWT)."""
        dwt_bit_length, lsb_bit_length = bit_lengths

        # 1. Ekstrak LSB
        print(f"Hybrid: Mengekstrak {lsb_bit_length} bit via LSB...")
        lsb_message_part = self.lsb_steganography.extract(stego_image, lsb_bit_length)

        # 2. Ekstrak DWT
        print(f"Hybrid: Mengekstrak {dwt_bit_length} bit via DWT...")
        dwt_message_part = self.dwt_steganography.extract(stego_image, dwt_bit_length)

        return dwt_message_part + lsb_message_part
    
DWT_LSB_DEFAULT_PARAM = {
    'dwt_lsb_ratio': (0.5, 0.5),
    'dwt_params': {'wavelet': 'haar', 'level': 3, 'band': 'HH', 'embed_level': 3, 'delta': 25.0, 'robust_mode': False},
    'lsb_params': {'bits_per_channel': 1}
}