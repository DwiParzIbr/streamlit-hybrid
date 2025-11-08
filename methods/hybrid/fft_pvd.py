from methods.frequency.fft import FFTSteganography
from methods.spatial.pvd import PVDSteganography
from helpers.message_binary import message_to_binary, binary_to_message

class FFTPVDHybrid:
    """
    Menggabungkan steganografi FFT dan PVD.
    """
    def __init__(self, fft_pvd_ratio=(0.5, 0.5), fft_params=None, pvd_params=None):
        """
        Inisialisasi metode hibrida.
        """
        if sum(fft_pvd_ratio) != 1.0:
            raise ValueError("Jumlah fft_pvd_ratio harus 1.0")

        self.fft_ratio = fft_pvd_ratio[0]
        self.pvd_ratio = fft_pvd_ratio[1]

        self.fft_steganography = FFTSteganography(**(fft_params or {}))
        self.pvd_steganography = PVDSteganography(**(pvd_params or {}))

    def embed(self, cover_image, secret_message):
        """Menyisipkan pesan rahasia menggunakan metode hibrida FFT-PVD."""

        # 1. Bagi pesan
        full_binary_message = message_to_binary(secret_message)
        split_point = int(len(full_binary_message) * self.fft_ratio)

        fft_message_part = binary_to_message(full_binary_message[:split_point])
        pvd_message_part = binary_to_message(full_binary_message[split_point:])

        # 2. Sisipkan FFT (Input: RGB, Output: RGB)
        print(f"Hybrid: Menyisipkan {len(full_binary_message[:split_point])} bit via FFT...")
        intermediate_stego, fft_bit_length = self.fft_steganography.embed(
            cover_image.copy(), fft_message_part
        )

        # 3. Sisipkan PVD (Input: RGB, Output: RGB)
        print(f"Hybrid: Menyisipkan {len(full_binary_message[split_point:])} bit via PVD...")
        final_stego, pvd_bit_length = self.pvd_steganography.embed(
            intermediate_stego, pvd_message_part
        )

        return final_stego, (fft_bit_length, pvd_bit_length)

    def extract(self, stego_image, bit_lengths):
        """Mengekstrak pesan rahasia (PVD dulu, baru FFT)."""
        fft_bit_length, pvd_bit_length = bit_lengths

        # 1. Ekstrak PVD
        print(f"Hybrid: Mengekstrak {pvd_bit_length} bit via PVD...")
        pvd_message_part = self.pvd_steganography.extract(stego_image, pvd_bit_length)

        # 2. Ekstrak FFT
        print(f"Hybrid: Mengekstrak {fft_bit_length} bit via FFT...")
        fft_message_part = self.fft_steganography.extract(stego_image, fft_bit_length)

        if fft_message_part is None:
            print("Peringatan: Ekstraksi FFT gagal, pesan mungkin tidak lengkap.")
            fft_message_part = ""

        return fft_message_part + pvd_message_part
        
        
FFT_PVD_DEFAULT_PARAM = {
    'fft_pvd_ratio': (0.5, 0.5),
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
    'pvd_params': {}
}