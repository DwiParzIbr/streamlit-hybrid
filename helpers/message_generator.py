import string

def create_pattern_padded_message(base_message: str, target_byte_size: int) -> str:
    """
    Pads a message with a repeating, deterministic pattern of characters
    to reach a specific byte size.

    Args:
        base_message (str): The initial message string.
        target_byte_size (int): The desired total size in bytes.

    Returns:
        str: The padded message string, or an empty string if the base
             message is already larger than the target size.
    """
    # 1. Define the pool of characters to create the repeating pattern.
    # These are all single-byte characters.
    character_pool = string.ascii_letters + string.digits + string.punctuation

    # 2. Encode the base message to find its current byte size.
    try:
        message_bytes = base_message.encode('utf-8')
    except UnicodeEncodeError as e:
        print(f"Error encoding the message: {e}")
        return ""

    current_byte_size = len(message_bytes)

    # 3. Check if the message is already too large.
    if current_byte_size > target_byte_size:
        print(f"⚠️ Warning: Base message is {current_byte_size} bytes, "
              f"which is larger than the target of {target_byte_size} bytes.")
        return ""

    # 4. Calculate the needed padding length.
    padding_needed = target_byte_size - current_byte_size

    # 5. Build the deterministic padding string.
    # This creates a long repeating pattern (e.g., "abcabcabc...") and
    # then cuts it to the exact length needed.
    pattern_repeats = (padding_needed // len(character_pool)) + 1
    full_pattern = character_pool * pattern_repeats
    padding = full_pattern[:padding_needed]

    return base_message + padding

# --- USAGE EXAMPLE ---
DEFAULT_MESSAGE = (
    "This message uses deterministic padding. 🚀 "
    "The padding is a repeating, non-random sequence. "
    "Let's test it: 123!@#<>? éçñ. 😊👍"
)

DEFAULT_TARGET_SIZE = 512

# Create the final message using the new function
# SECRET_MESSAGE = create_pattern_padded_message(DEFAULT_MESSAGE, DEFAULT_TARGET_SIZE)
