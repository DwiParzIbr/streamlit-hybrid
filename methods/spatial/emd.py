import numpy as np
from helpers.message_binary import message_to_binary, binary_to_message

class EMDSteganography:
    """
    EMD Steganography - RGB-Only Version.
    Assumes all input images are 3-channel (H, W, 3) NumPy arrays.
    """
    def __init__(self, n=2):
        self.n = n  # Number of pixels in a group (e.g., 2 for base-5)
        self.base = 2 * self.n + 1
        print(f"EMD Steganography initialized with n={n} (base-{self.base})")

    def _calculate_capacity(self, image: np.ndarray) -> int:
        """Calculates the max number of base-5 digits for an RGB image."""
        # Assume image shape is (H, W, 3)
        height, width, channels = image.shape

        if channels < 3:
             raise ValueError("Image must have at least 3 channels (RGB).")

        groups_per_channel = height * (width // self.n)
        return groups_per_channel * 3 # Explicitly use 3 channels

    def _extraction_function(self, pixels: list) -> int:
        """Extraction function for EMD: f(p1, ..., pn) = Σ(i * pi) mod (2n+1)"""
        return sum((i + 1) * int(pixels[i]) for i in range(self.n)) % self.base

    def _embed_digit(self, pixels: list, digit: int) -> list:
        """Embeds a single digit into a pixel group using a minimal distortion search."""
        current_digit = self._extraction_function(pixels)
        if current_digit == digit:
            return pixels.copy()

        # Search for a modification with the smallest change (+1 or -1)
        for i in range(self.n):
            for delta in [1, -1]:
                modified_pixels = pixels.copy()
                new_value = modified_pixels[i] + delta
                if 0 <= new_value <= 255:
                    modified_pixels[i] = new_value
                    if self._extraction_function(modified_pixels) == digit:
                        return modified_pixels

        # Try a slightly larger jump if +/- 1 fails
        for i in range(self.n):
            for delta in [2, -2]:
                 modified_pixels = pixels.copy()
                 new_value = modified_pixels[i] + delta
                 if 0 <= new_value <= 255:
                    modified_pixels[i] = new_value
                    if self._extraction_function(modified_pixels) == digit:
                        return modified_pixels

        # If still no solution, return the original pixels.
        # This will cause a BER error, but avoids a massive performance hit.
        return pixels

    def embed(self, cover_image: np.ndarray, secret_message: str) -> tuple:
        """Embeds a secret message into an RGB cover image using EMD."""
        # No grayscale check needed, assume (H, W, 3)

        stego_image = cover_image.copy().astype(np.int16)
        binary_message = message_to_binary(secret_message)
        bit_length = len(binary_message)

        # Convert binary message to base-n digits.
        # For n=2, base=5. We can store 2 bits (0-3) per digit.
        num_bits_per_digit = 2
        digits = []
        for i in range(0, len(binary_message), num_bits_per_digit):
            bits = binary_message[i:i + num_bits_per_digit]
            bits = bits.ljust(num_bits_per_digit, '0') # Pad last chunk
            digits.append(int(bits, 2)) # Values will be in {0, 1, 2, 3}

        max_capacity_digits = self._calculate_capacity(stego_image)
        if len(digits) > max_capacity_digits:
            raise ValueError(f"Message too long! Needs {len(digits)} groups, but capacity is {max_capacity_digits}.")

        height, width, _ = stego_image.shape
        digit_index = 0

        # Explicitly loop over 3 channels (R, G, B)
        for channel in range(3):
            for row in range(height):
                for col in range(0, width - self.n + 1, self.n):
                    if digit_index >= len(digits): break

                    pixel_group = [stego_image[row, col + i, channel] for i in range(self.n)]
                    new_pixel_group = self._embed_digit(pixel_group, digits[digit_index])

                    for i in range(self.n):
                        stego_image[row, col + i, channel] = new_pixel_group[i]
                    digit_index += 1
                if digit_index >= len(digits): break
            if digit_index >= len(digits): break

        final_stego_image = np.clip(stego_image, 0, 255).astype(np.uint8)
        return final_stego_image, bit_length

    def extract(self, stego_image: np.ndarray, bit_length: int) -> str:
        """Extracts a secret message from an RGB stego image."""
        # No grayscale check needed, assume (H, W, 3)

        height, width, _ = stego_image.shape
        num_bits_per_digit = 2 # Must match embed
        extracted_bits = ""

        digits_needed = (bit_length + num_bits_per_digit - 1) // num_bits_per_digit

        digit_count = 0
        # Explicitly loop over 3 channels (R, G, B)
        for channel in range(3):
            for row in range(height):
                for col in range(0, width - self.n + 1, self.n):
                    if digit_count >= digits_needed: break

                    pixel_group = [stego_image[row, col + i, channel] for i in range(self.n)]
                    extracted_digit = self._extraction_function(pixel_group)

                    # Only digits 0-3 were used for embedding
                    if extracted_digit < 4:
                        extracted_bits += format(extracted_digit, f'0{num_bits_per_digit}b')
                    else:
                        # Digit is 4, which was unused. This is an error.
                        # Add '00' as a placeholder to be caught by BER.
                        extracted_bits += "00"

                    digit_count += 1
                if digit_count >= digits_needed: break
            if digit_count >= digits_needed: break

        final_binary_string = extracted_bits[:bit_length]
        return binary_to_message(final_binary_string)
    
EMD_DEFAULT_PARAM = {'n': 2}