from methods.frequency.dwt import DWTSteganography
from methods.spatial.pvd import PVDSteganography
from helpers.message_binary import message_to_binary, binary_to_message

class DWTPVDHybrid:
    """
    Menggabungkan steganografi DWT dan PVD.
    """
    def __init__(self, dwt_pvd_ratio=(0.5, 0.5), dwt_params=None, pvd_params=None):
        """
        Inisialisasi metode hibrida.
        """
        if sum(dwt_pvd_ratio) != 1.0:
            raise ValueError("Jumlah dwt_pvd_ratio harus 1.0")

        self.dwt_ratio = dwt_pvd_ratio[0]
        self.pvd_ratio = dwt_pvd_ratio[1]

        self.dwt_steganography = DWTSteganography(**(dwt_params or {}))
        self.pvd_steganography = PVDSteganography(**(pvd_params or {}))

    def embed(self, cover_image, secret_message):
        """Menyisipkan pesan rahasia menggunakan metode hibrida DWT-PVD."""

        # 1. Bagi pesan
        full_binary_message = message_to_binary(secret_message)
        split_point = int(len(full_binary_message) * self.dwt_ratio)

        dwt_message_part = binary_to_message(full_binary_message[:split_point])
        pvd_message_part = binary_to_message(full_binary_message[split_point:])

        # 2. Sisipkan DWT (Input: RGB, Output: RGB)
        print(f"Hybrid: Menyisipkan {len(full_binary_message[:split_point])} bit via DWT...")
        intermediate_stego, dwt_bit_length = self.dwt_steganography.embed(
            cover_image.copy(), dwt_message_part
        )

        # 3. Sisipkan PVD (Input: RGB, Output: RGB)
        print(f"Hybrid: Menyisipkan {len(full_binary_message[split_point:])} bit via PVD...")
        final_stego, pvd_bit_length = self.pvd_steganography.embed(
            intermediate_stego, pvd_message_part
        )

        return final_stego, (dwt_bit_length, pvd_bit_length)

    def extract(self, stego_image, bit_lengths):
        """Mengekstrak pesan rahasia (PVD dulu, baru DWT)."""
        dwt_bit_length, pvd_bit_length = bit_lengths

        # 1. Ekstrak PVD
        print(f"Hybrid: Mengekstrak {pvd_bit_length} bit via PVD...")
        pvd_message_part = self.pvd_steganography.extract(stego_image, pvd_bit_length)

        # 2. Ekstrak DWT
        print(f"Hybrid: Mengekstrak {dwt_bit_length} bit via DWT...")
        dwt_message_part = self.dwt_steganography.extract(stego_image, dwt_bit_length)

        return dwt_message_part + pvd_message_part
    
DWT_PVD_DEFAULT_PARAM = {
    'dwt_pvd_ratio': (0.5, 0.5),
    'dwt_params': {'wavelet': 'haar', 'level': 3, 'band': 'HH', 'embed_level': 3, 'delta': 25.0, 'robust_mode': False},
    'pvd_params': {}
}