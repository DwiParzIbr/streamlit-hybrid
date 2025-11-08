from methods.frequency.fft import FFTSteganography
from methods.spatial.lsb import LSBSteganography
from helpers.message_binary import message_to_binary, binary_to_message

class FFTLSBHybrid:
    """
    Menggabungkan steganografi FFT dan LSB.
    """
    def __init__(self, fft_lsb_ratio=(0.5, 0.5), fft_params=None, lsb_params=None):
        """
        Inisialisasi metode hibrida.
        """
        if sum(fft_lsb_ratio) != 1.0:
            raise ValueError("Jumlah fft_lsb_ratio harus 1.0")

        self.fft_ratio = fft_lsb_ratio[0]
        self.lsb_ratio = fft_lsb_ratio[1]

        self.fft_steganography = FFTSteganography(**(fft_params or {}))
        self.lsb_steganography = LSBSteganography(**(lsb_params or {}))

    def embed(self, cover_image, secret_message):
        """Menyisipkan pesan rahasia menggunakan metode hibrida FFT-LSB."""

        # 1. Bagi pesan
        full_binary_message = message_to_binary(secret_message)
        split_point = int(len(full_binary_message) * self.fft_ratio)

        fft_message_part = binary_to_message(full_binary_message[:split_point])
        lsb_message_part = binary_to_message(full_binary_message[split_point:])

        # 2. Sisipkan FFT (Input: RGB, Output: RGB)
        print(f"Hybrid: Menyisipkan {len(full_binary_message[:split_point])} bit via FFT...")
        intermediate_stego, fft_bit_length = self.fft_steganography.embed(
            cover_image.copy(), fft_message_part
        )

        # 3. Sisipkan LSB (Input: RGB, Output: RGB)
        print(f"Hybrid: Menyisipkan {len(full_binary_message[split_point:])} bit via LSB...")
        final_stego, lsb_bit_length = self.lsb_steganography.embed(
            intermediate_stego, lsb_message_part
        )

        return final_stego, (fft_bit_length, lsb_bit_length)

    def extract(self, stego_image, bit_lengths):
        """Mengekstrak pesan rahasia (LSB dulu, baru FFT)."""
        fft_bit_length, lsb_bit_length = bit_lengths

        # 1. Ekstrak LSB
        print(f"Hybrid: Mengekstrak {lsb_bit_length} bit via LSB...")
        lsb_message_part = self.lsb_steganography.extract(stego_image, lsb_bit_length)

        # 2. Ekstrak FFT
        print(f"Hybrid: Mengekstrak {fft_bit_length} bit via FFT...")
        fft_message_part = self.fft_steganography.extract(stego_image, fft_bit_length)

        if fft_message_part is None:
            print("Peringatan: Ekstraksi FFT gagal, pesan mungkin tidak lengkap.")
            fft_message_part = "" # Kembalikan string kosong jika ekstraksi gagal

        return fft_message_part + lsb_message_part
    
FFT_LSB_DEFAULT_PARAM = {
    'fft_lsb_ratio': (0.5, 0.5),
    'fft_params': {
        'r_in': 0.1,
        'r_out': 0.4,
        'header_repeat': 3,
        'payload_repeat': 3,
        'header_channel': 'Cr',
        'payload_channel': 'Cb',
        'mag_min_boost': 3.0,
        'color_order': 'RGB'  # <-- Ditambahkan untuk konsistensi
    },
    'lsb_params': {'bits_per_channel': 1}
}