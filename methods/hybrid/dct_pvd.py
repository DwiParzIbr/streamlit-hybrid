from methods.frequency.dct import DCTSteganography
from methods.spatial.pvd import PVDSteganography
from helpers.message_binary import message_to_binary, binary_to_message
from methods.frequency.dct import DCT_POSITION_MID

class DCTPVDHybrid:
    """
    Menggabungkan steganografi DCT dan PVD.
    """
    def __init__(self, dct_pvd_ratio=(0.5, 0.5), dct_params=None, pvd_params=None):
        """
        Inisialisasi metode hibrida.
        """
        if sum(dct_pvd_ratio) != 1.0:
            raise ValueError("Jumlah dct_pvd_ratio harus 1.0")

        self.dct_ratio = dct_pvd_ratio[0]
        self.pvd_ratio = dct_pvd_ratio[1]

        # Inisialisasi kelas dasar
        self.dct_steganography = DCTSteganography(**(dct_params or {}))
        self.pvd_steganography = PVDSteganography(**(pvd_params or {}))

    def embed(self, cover_image, secret_message):
        """Menyisipkan pesan rahasia menggunakan metode hibrida DCT-PVD."""

        # 1. Bagi pesan
        full_binary_message = message_to_binary(secret_message)
        total_bits = len(full_binary_message)
        split_point = int(total_bits * self.dct_ratio)

        dct_binary_part = full_binary_message[:split_point]
        pvd_binary_part = full_binary_message[split_point:]

        dct_message_part = binary_to_message(dct_binary_part)
        pvd_message_part = binary_to_message(pvd_binary_part)

        # 2. Sisipkan DCT (Input: RGB, Output: RGB)
        print(f"Hybrid: Menyisipkan {len(dct_binary_part)} bit via DCT...")
        intermediate_stego, dct_bit_length = self.dct_steganography.embed(
            cover_image.copy(), dct_message_part
        )

        # 3. Sisipkan PVD (Input: RGB, Output: RGB)
        print(f"Hybrid: Menyisipkan {len(pvd_binary_part)} bit via PVD...")
        final_stego, pvd_bit_length = self.pvd_steganography.embed(
            intermediate_stego, pvd_message_part
        )

        return final_stego, (dct_bit_length, pvd_bit_length)

    def extract(self, stego_image, bit_lengths):
        """Mengekstrak pesan rahasia (PVD dulu, baru DCT)."""
        dct_bit_length, pvd_bit_length = bit_lengths

        # 1. Ekstrak PVD
        print(f"Hybrid: Mengekstrak {pvd_bit_length} bit via PVD...")
        pvd_message_part = self.pvd_steganography.extract(stego_image, pvd_bit_length)

        # 2. Ekstrak DCT
        print(f"Hybrid: Mengekstrak {dct_bit_length} bit via DCT...")
        dct_message_part = self.dct_steganography.extract(stego_image, dct_bit_length)

        # 3. Gabungkan
        return dct_message_part + pvd_message_part
    
DCT_PVD_DEFAULT_PARAM = {
    'dct_pvd_ratio': (0.5, 0.5),
    'dct_params': {'quant_factor': 70, 'embed_positions': DCT_POSITION_MID},
    'pvd_params': {}
}