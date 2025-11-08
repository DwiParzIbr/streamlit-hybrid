from methods.frequency.fft import FFTSteganography
from methods.spatial.emd import EMDSteganography
from helpers.message_binary import message_to_binary, binary_to_message

class FFTEMDHybrid:
    """
    Menggabungkan steganografi FFT dan EMD.
    """
    def __init__(self, fft_emd_ratio=(0.5, 0.5), fft_params=None, emd_params=None):
        """
        Inisialisasi metode hibrida.
        """
        if sum(fft_emd_ratio) != 1.0:
            raise ValueError("Jumlah fft_emd_ratio harus 1.0")

        self.fft_ratio = fft_emd_ratio[0]
        self.emd_ratio = fft_emd_ratio[1]

        self.fft_steganography = FFTSteganography(**(fft_params or {}))
        self.emd_steganography = EMDSteganography(**(emd_params or {}))

    def embed(self, cover_image, secret_message):
        """Menyisipkan pesan rahasia menggunakan metode hibrida FFT-EMD."""

        # 1. Bagi pesan
        full_binary_message = message_to_binary(secret_message)
        split_point = int(len(full_binary_message) * self.fft_ratio)

        fft_message_part = binary_to_message(full_binary_message[:split_point])
        emd_message_part = binary_to_message(full_binary_message[split_point:])

        # 2. Sisipkan FFT (Input: RGB, Output: RGB)
        print(f"Hybrid: Menyisipkan {len(full_binary_message[:split_point])} bit via FFT...")
        intermediate_stego, fft_bit_length = self.fft_steganography.embed(
            cover_image.copy(), fft_message_part
        )

        # 3. Sisipkan EMD (Input: RGB, Output: RGB)
        print(f"Hybrid: Menyisipkan {len(full_binary_message[split_point:])} bit via EMD...")
        final_stego, emd_bit_length = self.emd_steganography.embed(
            intermediate_stego, emd_message_part
        )

        return final_stego, (fft_bit_length, emd_bit_length)

    def extract(self, stego_image, bit_lengths):
        """Mengekstrak pesan rahasia (EMD dulu, baru FFT)."""
        fft_bit_length, emd_bit_length = bit_lengths

        # 1. Ekstrak EMD
        print(f"Hybrid: Mengekstrak {emd_bit_length} bit via EMD...")
        emd_message_part = self.emd_steganography.extract(stego_image, emd_bit_length)

        # 2. Ekstrak FFT
        print(f"Hybrid: Mengekstrak {fft_bit_length} bit via FFT...")
        fft_message_part = self.fft_steganography.extract(stego_image, fft_bit_length)

        if fft_message_part is None:
            print("Peringatan: Ekstraksi FFT gagal, pesan mungkin tidak lengkap.")
            fft_message_part = ""

        return fft_message_part + emd_message_part
        
FFT_EMD_DEFAULT_PARAM = {
    'fft_emd_ratio': (0.5, 0.5),
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
    'emd_params': {'n': 2}
}