import zlib
from typing import Tuple, List
import numpy as np
import cv2  # cv2 IS REQUIRED for YCrCb color space conversion
from numpy.fft import fft2, ifft2
from helpers.message_binary import message_to_binary, binary_to_message

# Assumes message_to_binary() and binary_to_message() exist

class FFTSteganography:
    """
    QIM on phase (4-bin) - RGB-Only Version.
    This class assumes all inputs are 3-channel NumPy arrays.
    It still REQUIRES cv2 for RGB <-> YCrCb color space conversion.
    """
    VERSION = 1
    HDR_VER_BITS   = 8
    HDR_SEED_BITS  = 32
    HDR_LEN_BITS   = 32
    HDR_CRC_BITS   = 32
    HDR_BITS = HDR_VER_BITS + HDR_SEED_BITS + HDR_LEN_BITS + HDR_CRC_BITS

    def __init__(self,
                 r_in: float = 0.30,
                 r_out: float = 0.50,
                 phase_levels: int = 4,
                 header_repeat: int = 5,
                 payload_repeat: int = 3,
                 color_order: str = "RGB",
                 header_channel: str = "Y",
                 payload_channel: str = "Cb",
                 mag_min_boost: float = 3.0):
        if phase_levels != 4:
            raise ValueError("phase_levels is locked to 4.")
        if header_repeat % 2 == 0:
            header_repeat += 1
        if payload_repeat % 2 == 0:
            payload_repeat += 1

        self.r_in = float(r_in)
        self.r_out = float(r_out)
        self.phase_levels = phase_levels
        self.HDR_R = int(header_repeat)
        self.PAY_R = int(payload_repeat)
        self.color_order = color_order.upper()
        if self.color_order not in ("RGB","BGR"):
            raise ValueError("color_order must be 'RGB' or 'BGR'.")
        self.header_channel  = header_channel
        self.payload_channel = payload_channel
        self.mag_min_boost = float(mag_min_boost)

    # ----- Internal utils (static methods remain the same) -----
    @staticmethod
    def _bytes_to_bits(b: bytes) -> str:
        return ''.join(f'{x:08b}' for x in b)
    @staticmethod
    def _bits_to_bytes(bits: str) -> bytes:
        n = len(bits) - (len(bits) % 8)
        return bytes(int(bits[i:i+8], 2) for i in range(0, n, 8))
    @staticmethod
    def _int_to_bits(v: int, k: int) -> str:
        return format(int(v) & ((1 << k) - 1), f'0{k}b')
    @staticmethod
    def _wrap_to_pi(a: float) -> float:
        return ((a + np.pi) % (2*np.pi)) - np.pi
    @staticmethod
    def _phase_to_bit(phase: float, phase_levels: int) -> int:
        phase_mod = phase % (2*np.pi)
        bin_size = 2*np.pi / phase_levels
        bin_idx = int(np.floor(phase_mod / bin_size)) % phase_levels
        return bin_idx & 1
    @staticmethod
    def _conj_partner(y, x, H, W):
        cy, cx = H//2, W//2
        return (2*cy - y) % H, (2*cx - x) % W
    # -----------------------------------------------------------

    def _annulus_positions(self, H: int, W: int) -> List[Tuple[int,int]]:
        cy, cx = H//2, W//2
        Y, X = np.ogrid[:H, :W]
        dist = np.sqrt((Y-cy)**2 + (X-cx)**2)
        R = min(H,W)/2.0
        mask = (dist >= self.r_in*R) & (dist <= self.r_out*R)
        ys, xs = np.where(mask)
        items = []
        for y, x in zip(ys, xs):
            if y >= cy or x == cx:
                continue
            r_int = int(round(dist[y,x]))
            ang = float(np.arctan2(y-cy, x-cx))
            items.append((r_int, ang, int(y), int(x)))
        items.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
        return [(y,x) for _,_,y,x in items]

    def _split_ycrcb(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        MODIFIED: Removed grayscale check. Assumes 3-channel input.
        """
        if self.color_order == "RGB":
            ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        return y.astype(np.float64), cr.astype(np.float64), cb.astype(np.float64)

    def _merge_ycrcb(self, y_u8: np.ndarray, cr_f: np.ndarray, cb_f: np.ndarray) -> np.ndarray:
        """
        MODIFIED: Removed grayscale check.
        """
        ycrcb = cv2.merge([y_u8, cr_f.astype(np.uint8), cb_f.astype(np.uint8)])
        if self.color_order == "RGB":
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        else:
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    def _write_grouped_bits(self, plane_f64: np.ndarray, bits: str, repeat: int, allow_boost: bool):
        # (This method's logic remains unchanged)
        H, W = plane_f64.shape
        F = np.fft.fftshift(fft2(plane_f64))
        mag, phase = np.abs(F), np.angle(F)
        positions = self._annulus_positions(H, W)
        need = len(bits) * repeat
        if need > len(positions):
            raise ValueError(f"Plane capacity insufficient: need {need}, have {len(positions)}.")
        groups = [ positions[i*repeat:(i+1)*repeat] for i in range(len(bits)) ]

        bin_size = 2*np.pi / self.phase_levels
        centers = np.array([(k+0.5)*bin_size for k in range(self.phase_levels)], dtype=np.float64)
        mag_floor = float(self.mag_min_boost) if allow_boost else 0.0

        def ang_dist(a, b):
            d = (a - b + np.pi) % (2*np.pi) - np.pi
            return abs(d)

        def write_bit_at(yx, bit):
            y0, x0 = yx
            if allow_boost and mag[y0, x0] < mag_floor:
                mag[y0, x0] = mag_floor
            ph = phase[y0, x0]
            cand = [0,2] if bit==0 else [1,3]
            norm = ph if ph >= 0 else ph + 2*np.pi
            target = min((centers[j] for j in cand), key=lambda c: ang_dist(norm, c))
            phase[y0, x0] = self._wrap_to_pi(target)
            ys, xs = self._conj_partner(y0, x0, H, W)
            if 0<=ys<H and 0<=xs<W and (ys!=y0 or xs!=x0):
                if allow_boost and mag[ys, xs] < mag_floor:
                    mag[ys, xs] = mag_floor
                phase[ys, xs] = -phase[y0, x0]

        for i, grp in enumerate(groups):
            b = int(bits[i])
            for yx in grp:
                write_bit_at(yx, b)

        F_new = (mag * np.exp(1j*phase))
        plane_new = np.real(ifft2(np.fft.ifftshift(F_new)))
        return np.clip(plane_new, 0, 255).astype(np.uint8)

    def _read_grouped_bits(self, plane_u8: np.ndarray, bit_count: int, repeat: int) -> str:
        # (This method's logic remains unchanged)
        plane_f64 = plane_u8.astype(np.float64)
        H, W = plane_f64.shape
        F = np.fft.fftshift(fft2(plane_f64))
        phase = np.angle(F)
        positions = self._annulus_positions(H, W)
        need = bit_count * repeat
        if need > len(positions):
            raise ValueError("Payload larger than capacity.")
        groups = [ positions[i*repeat:(i+1)*repeat] for i in range(bit_count) ]
        out = []
        for grp in groups:
            votes = sum(self._phase_to_bit(phase[y,x], self.phase_levels) for y,x in grp)
            out.append('1' if votes > (len(grp)//2) else '0')
        return ''.join(out)

    def embed(self, cover_image: np.ndarray, secret_message: str):
        """Embeds a secret message into a cover image using FFT."""
        # Assumes cover_image is 3-channel (RGB or BGR per self.color_order)
        y_f, cr_f, cb_f = self._split_ycrcb(cover_image)
        H, W = y_f.shape
        pos_count = len(self._annulus_positions(H, W))
        payload_capacity_bits = pos_count // self.PAY_R

        pay_bits = message_to_binary(secret_message)
        bit_length = len(pay_bits)

        if bit_length > payload_capacity_bits:
            raise ValueError(f"Message too long. Capacity: {payload_capacity_bits} bits, Needed: {bit_length} bits.")

        seed = int(np.random.default_rng().integers(0, 2**32-1, dtype=np.uint32))
        payload = secret_message.encode('utf-8')
        crc = zlib.crc32(payload) & 0xFFFFFFFF

        header_bits = (self._int_to_bits(self.VERSION, self.HDR_VER_BITS) +
                       self._int_to_bits(seed, self.HDR_SEED_BITS) +
                       self._int_to_bits(bit_length, self.HDR_LEN_BITS) +
                       self._int_to_bits(crc, self.HDR_CRC_BITS))

        rng = np.random.default_rng(int(seed))
        perm = rng.permutation(len(pay_bits))
        pay_bits_perm = ''.join(pay_bits[i] for i in perm)

        def get_plane(name):
            # MODIFIED: Removed grayscale checks
            if name == "Y": return y_f
            if name == "Cr": return cr_f
            if name == "Cb": return cb_f
            raise ValueError(f"Unknown channel name: {name}")

        def set_plane(name, new_u8):
            # MODIFIED: Removed grayscale checks
            nonlocal y_f, cr_f, cb_f
            if name == "Y": y_f = new_u8.astype(np.float64)
            elif name == "Cr": cr_f = new_u8.astype(np.float64)
            elif name == "Cb": cb_f = new_u8.astype(np.float64)

        # Embed payload first
        payload_plane = get_plane(self.payload_channel)
        payload_new = self._write_grouped_bits(payload_plane, pay_bits_perm, repeat=self.PAY_R, allow_boost=True)
        set_plane(self.payload_channel, payload_new)

        # Embed header last
        header_new = self._write_grouped_bits(get_plane(self.header_channel), header_bits, repeat=self.HDR_R, allow_boost=False)
        set_plane(self.header_channel, header_new)

        return self._merge_ycrcb(y_f.astype(np.uint8), cr_f, cb_f), bit_length

    def extract(self, stego_image: np.ndarray, bit_length: int = None) -> str | None:
        """Extracts a secret message using predefined bit length or from header."""
        # Assumes stego_image is 3-channel
        y_f, cr_f, cb_f = self._split_ycrcb(stego_image)

        def get_u8(name):
            # MODIFIED: Removed grayscale checks
            if name == "Y": return y_f.astype(np.uint8)
            if name == "Cr": return cr_f.astype(np.uint8)
            if name == "Cb": return cb_f.astype(np.uint8)
            raise ValueError(f"Unknown channel name: {name}")

        # Read header
        hp = get_u8(self.header_channel)
        hbits = self._read_grouped_bits(hp, self.HDR_BITS, self.HDR_R)
        try:
            v0 = int(hbits[0:self.HDR_VER_BITS], 2)
            seed = int(hbits[self.HDR_VER_BITS:self.HDR_VER_BITS+self.HDR_SEED_BITS], 2)
            length_bits = int(hbits[self.HDR_VER_BITS+self.HDR_SEED_BITS:self.HDR_VER_BITS+self.HDR_SEED_BITS+self.HDR_LEN_BITS], 2)
            crc_expected = int(hbits[-self.HDR_CRC_BITS:], 2)
        except Exception:
            return ""
        if v0 != self.VERSION:
            return ""

        if bit_length is None:
            bit_length = length_bits
        elif bit_length != length_bits:
            print(f"Warning: Provided bit_length ({bit_length}) doesn't match header ({length_bits}). Using header value.")
            bit_length = length_bits

        if bit_length == 0:
            return "" # Handle empty message case

        # Read payload
        pp = get_u8(self.payload_channel)
        bits_perm = self._read_grouped_bits(pp, bit_length, self.PAY_R)

        # Invert permutation
        rng = np.random.default_rng(int(seed))
        perm = rng.permutation(bit_length)
        inv_perm = np.zeros_like(perm)
        inv_perm[perm] = np.arange(len(perm))

        bits_arr = np.array(list(bits_perm))
        bits = ''.join(bits_arr[inv_perm])

        # Verify CRC
        payload = self._bits_to_bytes(bits)
        crc_actual = zlib.crc32(payload) & 0xFFFFFFFF
        if crc_actual != crc_expected:
            print(f"CRC mismatch: expected {crc_expected}, got {crc_actual}")
            
            return binary_to_message(bits)

        return binary_to_message(bits)
        
FFT_DEFAULT_PARAM = {
    'r_in': 0.1,
    'r_out': 0.4,
    'header_repeat': 3,
    'payload_repeat': 3,
    'header_channel': 'Cr',
    'payload_channel': 'Cb',
    'mag_min_boost': 3.0,
    'color_order': 'RGB' # <-- Explicitly set to 'RGB' to match our inputs
}