def message_to_binary(message: str) -> str:
    """
    Converts a UTF-8 string into a continuous binary string.

    Args:
        message: The input string.

    Returns:
        A string of '0's and '1's representing the message.
    """
    # Encode the string into bytes using the UTF-8 standard
    encoded_bytes = message.encode('utf-8')

    # Convert each byte to its 8-bit binary representation and join them
    binary_string = ''.join(format(byte, '08b') for byte in encoded_bytes)

    return binary_string

def binary_to_message(binary_str: str) -> str:
    """
    Converts a continuous binary string back into a UTF-8 string.

    Args:
        binary_str: A string of '0's and '1's.

    Returns:
        The decoded string.
    """
    # Ensure the binary string length is a multiple of 8
    if len(binary_str) % 8 != 0:
        print("Warning: Binary string length is not a multiple of 8. Data may be corrupt.")

    # Create a list of 8-bit chunks
    byte_chunks = [binary_str[i:i+8] for i in range(0, len(binary_str), 8)]

    # Convert each binary chunk to an integer, then build a bytes object
    byte_data = bytes([int(chunk, 2) for chunk in byte_chunks])

    # Decode the bytes object back into a string using UTF-8
    # 'errors="replace"' will insert a '' for any invalid byte sequences
    try:
        message = byte_data.decode('utf-8')
    except UnicodeDecodeError:
        print("Error: Could not decode binary string with standard UTF-8. Replacing invalid characters.")
        message = byte_data.decode('utf-8', errors='replace')

    return message