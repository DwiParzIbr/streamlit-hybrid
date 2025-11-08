from methods.frequency.dct import DCTSteganography
from methods.spatial.lsb import LSBSteganography
from helpers.message_binary import message_to_binary, binary_to_message
from methods.frequency.dct import DCT_POSITION_MID

class DCTLSBHybrid:
    """
    Menggabungkan steganografi DCT dan LSB.

    Pesan dibagi berdasarkan rasio dan disisipkan secara berurutan:
    1. Sebagian pesan disisipkan menggunakan DCT (lebih tahan banting).
    2. Sisa pesan disisipkan ke dalam gambar hasil DCT menggunakan LSB.
    """
    def __init__(self, dct_lsb_ratio=(0.5, 0.5), dct_params=None, lsb_params=None):
        """
        Inisialisasi metode hibrida.
        """
        if sum(dct_lsb_ratio) != 1.0:
            raise ValueError("Jumlah dct_lsb_ratio harus 1.0")

        self.dct_ratio = dct_lsb_ratio[0]
        self.lsb_ratio = dct_lsb_ratio[1]

        # Inisialisasi kelas dasar dengan parameter yang diberikan atau default
        self.dct_steganography = DCTSteganography(**(dct_params or {}))
        self.lsb_steganography = LSBSteganography(**(lsb_params or {}))

    def embed(self, cover_image, secret_message):
        """Menyisipkan pesan rahasia menggunakan metode hibrida."""

        # 1. Ubah seluruh pesan menjadi biner dan bagi sesuai rasio
        full_binary_message = message_to_binary(secret_message)
        total_bits = len(full_binary_message)
        split_point = int(total_bits * self.dct_ratio)

        dct_binary_part = full_binary_message[:split_point]
        lsb_binary_part = full_binary_message[split_point:]

        dct_message_part = binary_to_message(dct_binary_part)
        lsb_message_part = binary_to_message(lsb_binary_part)

        # 2. Lakukan penyisipan DCT terlebih dahulu
        # (Input: RGB, Output: RGB)
        print(f"Hybrid: Menyisipkan {len(dct_binary_part)} bit via DCT...")
        intermediate_stego, dct_bit_length = self.dct_steganography.embed(
            cover_image.copy(), dct_message_part
        )

        # 3. Gunakan gambar hasil DCT sebagai cover untuk penyisipan LSB
        # (Input: RGB, Output: RGB)
        print(f"Hybrid: Menyisipkan {len(lsb_binary_part)} bit via LSB...")
        final_stego, lsb_bit_length = self.lsb_steganography.embed(
            intermediate_stego, lsb_message_part
        )

        # 4. Kembalikan gambar akhir dan tuple berisi panjang bit
        return final_stego, (dct_bit_length, lsb_bit_length)

    def extract(self, stego_image, bit_lengths):
        """Mengekstrak pesan rahasia (LSB dulu, baru DCT)."""
        dct_bit_length, lsb_bit_length = bit_lengths

        # 1. Ekstrak bagian LSB terlebih dahulu dari gambar stego akhir
        print(f"Hybrid: Mengekstrak {lsb_bit_length} bit via LSB...")
        lsb_message_part = self.lsb_steganography.extract(stego_image, lsb_bit_length)

        # 2. Ekstrak bagian DCT dari gambar stego yang sama
        #    Ini adalah poin penting: Ekstraksi LSB tidak (seharusnya)
        #    merusak data DCT secara signifikan.
        print(f"Hybrid: Mengekstrak {dct_bit_length} bit via DCT...")
        dct_message_part = self.dct_steganography.extract(stego_image, dct_bit_length)

        # 3. Gabungkan kembali kedua bagian pesan
        return dct_message_part + lsb_message_part
    
DCT_LSB_DEFAULT_PARAM = {
    'dct_lsb_ratio': (0.5, 0.5),
    'dct_params': {'quant_factor': 70, 'embed_positions': DCT_POSITION_MID},
    'lsb_params': {'bits_per_channel': 1}
}