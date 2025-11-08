from methods.frequency.dwt import DWTSteganography
from methods.spatial.emd import EMDSteganography
from helpers.message_binary import message_to_binary, binary_to_message

class DWTEmdHybrid:
    """
    Menggabungkan steganografi DWT dan EMD.
    """
    def __init__(self, dwt_emd_ratio=(0.5, 0.5), dwt_params=None, emd_params=None):
        """
        Inisialisasi metode hibrida.
        """
        if sum(dwt_emd_ratio) != 1.0:
            raise ValueError("Jumlah dwt_emd_ratio harus 1.0")

        self.dwt_ratio = dwt_emd_ratio[0]
        self.emd_ratio = dwt_emd_ratio[1]

        self.dwt_steganography = DWTSteganography(**(dwt_params or {}))
        self.emd_steganography = EMDSteganography(**(emd_params or {}))

    def embed(self, cover_image, secret_message):
        """Menyisipkan pesan rahasia menggunakan metode hibrida DWT-EMD."""

        # 1. Bagi pesan
        full_binary_message = message_to_binary(secret_message)
        split_point = int(len(full_binary_message) * self.dwt_ratio)

        dwt_message_part = binary_to_message(full_binary_message[:split_point])
        emd_message_part = binary_to_message(full_binary_message[split_point:])

        # 2. Sisipkan DWT (Input: RGB, Output: RGB)
        print(f"Hybrid: Menyisipkan {len(full_binary_message[:split_point])} bit via DWT...")
        intermediate_stego, dwt_bit_length = self.dwt_steganography.embed(
            cover_image.copy(), dwt_message_part
        )

        # 3. Sisipkan EMD (Input: RGB, Output: RGB)
        print(f"Hybrid: Menyisipkan {len(full_binary_message[split_point:])} bit via EMD...")
        final_stego, emd_bit_length = self.emd_steganography.embed(
            intermediate_stego, emd_message_part
        )

        return final_stego, (dwt_bit_length, emd_bit_length)

    def extract(self, stego_image, bit_lengths):
        """Mengekstrak pesan rahasia (EMD dulu, baru DWT)."""
        dwt_bit_length, emd_bit_length = bit_lengths

        # 1. Ekstrak EMD
        print(f"Hybrid: Mengekstrak {emd_bit_length} bit via EMD...")
        emd_message_part = self.emd_steganography.extract(stego_image, emd_bit_length)

        # 2. Ekstrak DWT
        print(f"Hybrid: Mengekstrak {dwt_bit_length} bit via DWT...")
        dwt_message_part = self.dwt_steganography.extract(stego_image, dwt_bit_length)

        return dwt_message_part + emd_message_part
    
DWT_EMD_DEFAULT_PARAM = {
    'dwt_emd_ratio': (0.5, 0.5),
    'dwt_params': {'wavelet': 'haar', 'level': 3, 'band': 'HH', 'embed_level': 3, 'delta': 25.0, 'robust_mode': False},
    'emd_params': {'n': 2}
}