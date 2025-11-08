from methods.frequency.dct import DCTSteganography
from methods.spatial.emd import EMDSteganography
from helpers.message_binary import message_to_binary, binary_to_message
from methods.frequency.dct import DCT_POSITION_MID

class DCTEMDHybrid:
    """
    Menggabungkan steganografi DCT dan EMD.
    """
    def __init__(self, dct_emd_ratio=(0.5, 0.5), dct_params=None, emd_params=None):
        """
        Inisialisasi metode hibrida.
        """
        if sum(dct_emd_ratio) != 1.0:
            raise ValueError("Jumlah dct_emd_ratio harus 1.0")

        self.dct_ratio = dct_emd_ratio[0]
        self.emd_ratio = dct_emd_ratio[1]

        self.dct_steganography = DCTSteganography(**(dct_params or {}))
        self.emd_steganography = EMDSteganography(**(emd_params or {}))

    def embed(self, cover_image, secret_message):
        """Menyisipkan pesan rahasia menggunakan metode hibrida DCT-EMD."""

        # 1. Bagi pesan
        full_binary_message = message_to_binary(secret_message)
        split_point = int(len(full_binary_message) * self.dct_ratio)

        dct_message_part = binary_to_message(full_binary_message[:split_point])
        emd_message_part = binary_to_message(full_binary_message[split_point:])

        # 2. Sisipkan DCT (Input: RGB, Output: RGB)
        print(f"Hybrid: Menyisipkan {len(full_binary_message[:split_point])} bit via DCT...")
        intermediate_stego, dct_bit_length = self.dct_steganography.embed(
            cover_image.copy(), dct_message_part
        )

        # 3. Sisipkan EMD (Input: RGB, Output: RGB)
        print(f"Hybrid: Menyisipkan {len(full_binary_message[split_point:])} bit via EMD...")
        final_stego, emd_bit_length = self.emd_steganography.embed(
            intermediate_stego, emd_message_part
        )

        return final_stego, (dct_bit_length, emd_bit_length)

    def extract(self, stego_image, bit_lengths):
        """Mengekstrak pesan rahasia (EMD dulu, baru DCT)."""
        dct_bit_length, emd_bit_length = bit_lengths

        # 1. Ekstrak EMD
        print(f"Hybrid: Mengekstrak {emd_bit_length} bit via EMD...")
        emd_message_part = self.emd_steganography.extract(stego_image, emd_bit_length)

        # 2. Ekstrak DCT
        print(f"Hybrid: Mengekstrak {dct_bit_length} bit via DCT...")
        dct_message_part = self.dct_steganography.extract(stego_image, dct_bit_length)

        return dct_message_part + emd_message_part
    
DCT_EMD_DEFAULT_PARAM = {
    'dct_emd_ratio': (0.5, 0.5),  # 50% DCT, 50% EMD
    'dct_params': {'quant_factor': 70, 'embed_positions': DCT_POSITION_MID},
    'emd_params': {'n': 2}
}