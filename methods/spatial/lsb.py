import numpy as np
from helpers.message_binary import message_to_binary, binary_to_message


class LSBSteganography:
    def __init__(self, bits_per_channel=1):
        """
        Initializes the LSB steganography tool.

        Args:
            bits_per_channel (int): Number of LSBs to use in each color
                                    channel (1-4).
        """
        # Ensure bits_per_channel is within a reasonable range (1-4)
        self.bits_per_channel = max(1, min(bits_per_channel, 4))
        print(f"LSB Steganography initialized to use {self.bits_per_channel} bit(s) per channel.")

    def embed(self, cover_image: np.ndarray, secret_message: str) -> tuple:
        """
        Embeds a secret message into an RGB image.

        Args:
            cover_image (np.ndarray): The original RGB image (H, W, 3).
            secret_message (str): The string message to embed.

        Returns:
            tuple: (stego_image (np.ndarray), bit_length (int))
        """
        # We assume cover_image is already a 3-channel RGB ndarray
        stego_image = cover_image.copy()

        # Use the robust function to convert the message
        binary_message = message_to_binary(secret_message)
        bit_length = len(binary_message)

        height, width, channels = stego_image.shape

        # We explicitly use 3 channels (RGB) for capacity calculation
        max_capacity = height * width * 3 * self.bits_per_channel

        if bit_length > max_capacity:
            raise ValueError(
                f"Message too long! Needs {bit_length} bits, but capacity is {max_capacity}."
            )

        binary_index = 0
        # Create a mask to clear the LSBs that we will modify
        # e.g., for 1 bit: 255 - 1 = 254 (11111110)
        # e.g., for 2 bits: 255 - 3 = 252 (11111100)
        clear_mask = 255 - (2**self.bits_per_channel - 1)

        for row in range(height):
            for col in range(width):
                # Only iterate over the 3 RGB channels
                for channel in range(3):
                    if binary_index >= bit_length:
                        # Break all loops if message is fully embedded
                        return stego_image, bit_length

                    pixel_value = int(stego_image[row, col, channel])

                    # 1. Clear the target LSBs of the pixel
                    pixel_value &= clear_mask

                    # 2. Get the next chunk of bits from the message
                    # Ensure we don't read past the end of the message
                    end_chunk_index = min(binary_index + self.bits_per_channel, bit_length)
                    bits_to_embed_str = binary_message[binary_index : end_chunk_index]

                    # Pad the chunk if it's at the very end and is shorter
                    bits_to_embed_str = bits_to_embed_str.ljust(self.bits_per_channel, '0')
                    embed_value = int(bits_to_embed_str, 2)

                    # 3. Combine the cleared pixel with the message bits
                    stego_image[row, col, channel] = pixel_value | embed_value

                    binary_index += self.bits_per_channel

        return stego_image, bit_length

    def extract(self, stego_image: np.ndarray, bit_length: int) -> str:
        """
        Extracts a secret message from an RGB stego image.

        Args:
            stego_image (np.ndarray): The stego image (H, W, 3).
            bit_length (int): The exact number of bits to extract.

        Returns:
            str: The extracted secret message.
        """
        # We assume stego_image is a 3-channel RGB ndarray
        height, width, channels = stego_image.shape

        # Create a mask to isolate the LSBs
        # e.g., for 1 bit: 1 (00000001)
        # e.g., for 2 bits: 3 (00000011)
        extract_mask = (2**self.bits_per_channel - 1)

        extracted_bits = ""
        bits_to_extract = bit_length

        for row in range(height):
            for col in range(width):
                # Only iterate over the 3 RGB channels
                for channel in range(3):
                    if len(extracted_bits) >= bits_to_extract:
                        # Break all loops once all bits are extracted
                        # Slice to the exact bit_length before decoding
                        final_binary_string = extracted_bits[:bits_to_extract]
                        return binary_to_message(final_binary_string)

                    pixel_value = int(stego_image[row, col, channel])

                    # 1. Isolate the LSBs
                    extracted_chunk_val = pixel_value & extract_mask

                    # 2. Format as a binary string, padding with leading zeros
                    extracted_chunk_str = format(extracted_chunk_val, f'0{self.bits_per_channel}b')

                    extracted_bits += extracted_chunk_str

        # Final processing after loops (in case bit_length was exact)
        final_binary_string = extracted_bits[:bits_to_extract]
        return binary_to_message(final_binary_string)
    
# Image.fromarray(stego_image).save(save_path)

LSB_DEFAULT_PARAM = {'bits_per_channel': 1}