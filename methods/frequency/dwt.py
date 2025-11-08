import numpy as np
import cv2
import pywt
from helpers.message_binary import message_to_binary, binary_to_message

class DWTSteganography:
    """
    Steganografi dalam domain DWT dengan penyisipan QIM (paritas) menggunakan bit_length logic.

    Fitur utama:
    - Parameter dapat disesuaikan sepenuhnya: wavelet, level dekomposisi, sub-band,
      level penyisipan, dan delta.
    - Mode Kuat (Robust Mode) yang dapat diaktifkan/nonaktifkan, dengan parameter
      repetisi dan interleaving yang dapat dikonfigurasi.
    - Menggunakan bit_length untuk menentukan akhir pesan, bukan delimiter.
    """
    def __init__(self, wavelet='haar', level=1, band='HL', delta: float | None = None,
                 embed_level=1, robust_mode=True, header_reps=5, payload_reps=3,
                 interleave_seed=1337, min_delta=5.0):
        """
        Inisialisasi objek DWTSteganography.

        Args:
            wavelet (str): Jenis wavelet (e.g., 'haar', 'db2', 'bior2.2').
            level (int): Jumlah level dekomposisi DWT.
            band (str): Sub-band target untuk penyisipan ('LL', 'LH', 'HL', 'HH').
            delta (float | None): Langkah kuantisasi. Jika None, akan dihitung secara adaptif.
            embed_level (int): Level dekomposisi tempat penyisipan dilakukan (1 <= embed_level <= level).
            robust_mode (bool): Jika True, aktifkan repetisi bit dan interleaving.
            header_reps (int): Jumlah repetisi untuk setiap bit header (jika robust_mode True).
            payload_reps (int): Jumlah repetisi untuk setiap bit payload (jika robust_mode True).
            interleave_seed (int): Seed untuk proses interleaving acak (jika robust_mode True).
            min_delta (float): Nilai delta minimum yang diizinkan saat dihitung secara adaptif.
        """
        # --- Parameter Transformasi ---
        self.wavelet = wavelet
        self.level = int(level)
        self.band = band.upper()
        self.delta = None if delta is None else float(delta)
        self.embed_level = int(embed_level)

        # --- Parameter Mode Kuat (Robust Mode) ---
        self.robust_mode = robust_mode
        self.header_reps = int(header_reps)
        self.payload_reps = int(payload_reps)
        self.interleave_seed = int(interleave_seed)
        self.min_delta = float(min_delta)

        # --- Validasi Input ---
        if self.band not in ['LL', 'LH', 'HL', 'HH']:
            raise ValueError("Band harus salah satu dari: 'LL', 'LH', 'HL', 'HH'")
        if self.level < 1 or self.embed_level < 1 or self.embed_level > self.level:
            raise ValueError(f"Level tidak valid: level={self.level}, embed_level={self.embed_level}. Syarat: 1 <= embed_level <= level.")
        pywt.Wavelet(self.wavelet) # Akan error jika wavelet tidak valid

    def _repeat_bits(self, s: str, r: int) -> str:
        if r <= 1: return s
        return ''.join(ch * r for ch in s)

    def _majority_decode(self, repeated: str, r: int) -> str:
        if r <= 1: return repeated
        return ''.join('1' if chunk.count('1') > len(chunk)/2 else '0' for i in range(0, len(repeated), r) if (chunk := repeated[i:i+r]))

    def _interleave(self, s: str, seed: int) -> str:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(s)); rng.shuffle(idx)
        return ''.join(np.array(list(s))[idx])

    def _deinterleave(self, s: str, seed: int) -> str:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(s)); rng.shuffle(idx)
        arr = np.empty_like(idx, dtype='<U1'); arr[idx] = list(s)
        return ''.join(arr)

    def _calculate_adaptive_delta(self, coeffs_flat: np.ndarray) -> float:
        coeffs_abs = np.abs(coeffs_flat)
        std = float(np.std(coeffs_abs))
        mean = float(np.mean(coeffs_abs))
        return float(np.clip(0.15 * std + 0.05 * mean, 2.0, 20.0))

    def _get_subband_ref(self, coeffs):
        if self.band == 'LL':
            subband, setter = coeffs[0], lambda new_arr: coeffs.__setitem__(0, new_arr)
        else:
            detail_tuple = coeffs[-self.embed_level]
            idx = {'HL': 0, 'LH': 1, 'HH': 2}[self.band]
            subband = detail_tuple[idx]
            def setter(new_arr):
                lst = list(coeffs[-self.embed_level])
                lst[idx] = new_arr
                coeffs[-self.embed_level] = tuple(lst)
        return subband, setter

    def calculate_capacity_bits(self, image_rgb: np.ndarray) -> int:
        ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
        coeffs = pywt.wavedec2(ycrcb[:, :, 0].astype(np.float32), self.wavelet, level=self.level)
        sub, _ = self._get_subband_ref(coeffs)
        return max(0, sub.size)

    def embed(self, cover_image: np.ndarray, secret_message: str):
        # Convert message to binary
        binary_message = message_to_binary(secret_message)
        bit_length = len(binary_message)

        if bit_length > self.calculate_capacity_bits(cover_image):
            raise ValueError(f"Pesan terlalu panjang! Kapasitas: {self.calculate_capacity_bits(cover_image)} bit, Dibutuhkan: {bit_length} bit.")

        ycrcb = cv2.cvtColor(cover_image, cv2.COLOR_RGB2YCrCb)
        y = ycrcb[:, :, 0].astype(np.float32)
        coeffs = pywt.wavedec2(y, self.wavelet, level=self.level)
        sub, setter = self._get_subband_ref(coeffs)
        sub_flat = sub.astype(np.float64).ravel()

        used_delta = self._calculate_adaptive_delta(sub_flat) if self.delta is None else self.delta
        used_delta = max(used_delta, self.min_delta)

        if self.robust_mode:
            data_bits = self._interleave(self._repeat_bits(binary_message, self.payload_reps), self.interleave_seed)
        else:
            data_bits = binary_message

        bitstream = np.fromiter(data_bits, dtype=np.int8)
        q = np.round(sub_flat[:len(bitstream)] / used_delta)
        mismatch = (q.astype(np.int64) & 1) != bitstream
        q[mismatch] += np.where(q[mismatch] >= 0, 1.0, -1.0)
        sub_flat[:len(bitstream)] = q * used_delta

        setter(sub_flat.reshape(sub.shape))
        y_rec = pywt.waverec2(coeffs, self.wavelet)[:y.shape[0], :y.shape[1]]
        ycrcb[:, :, 0] = np.clip(y_rec, 0, 255).astype(np.uint8)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB), bit_length

    def extract(self, stego_image: np.ndarray, bit_length: int):
        ycrcb = cv2.cvtColor(stego_image, cv2.COLOR_RGB2YCrCb)
        y = ycrcb[:, :, 0].astype(np.float32)
        coeffs = pywt.wavedec2(y, self.wavelet, level=self.level)
        sub, _ = self._get_subband_ref(coeffs)
        sub_flat = sub.astype(np.float64).ravel()

        if sub_flat.size < bit_length:
            return ""

        used_delta = self._calculate_adaptive_delta(sub_flat) if self.delta is None else self.delta
        used_delta = max(used_delta, self.min_delta)

        # Calculate the effective number of bits to extract based on robust mode
        effective_bit_length = bit_length * (self.payload_reps if self.robust_mode else 1)

        qa = np.round(sub_flat[:effective_bit_length] / used_delta).astype(np.int64)
        bits_payload = ''.join(map(str, qa & 1))

        if self.robust_mode:
            bits_payload = self._deinterleave(bits_payload, self.interleave_seed)
            bits_payload = self._majority_decode(bits_payload, self.payload_reps)

        # Trim the extracted bits to the exact original length
        final_binary_string = bits_payload[:bit_length]

        # Convert binary string to message
        return binary_to_message(final_binary_string)
    
DWT_DEFAULT_PARAM = {'wavelet': 'haar', 'level': 3, 'band': 'HH', 'embed_level': 3, 'delta': 25.0, 'robust_mode': False} # Added robust_mode=False