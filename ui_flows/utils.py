# ui_flows/utils.py

import streamlit as st
from helpers.message_generator import create_pattern_padded_message
from constants import DEFAULT_MESSAGE

def generate_dummy_message_callback(target_bit_len_key, message_key):
    """
    Fungsi callback yang aman untuk memodifikasi st.session_state 
    yang terikat pada st.text_area.
    """
    dummy_bit_len = st.session_state[target_bit_len_key]
    
    target_byte_size = (dummy_bit_len + 7) // 8 
    
    # Base message sekarang SELALU berasal dari DEFAULT_MESSAGE
    padded_msg = create_pattern_padded_message(DEFAULT_MESSAGE, target_byte_size)
    
    st.session_state[message_key] = padded_msg
    
    st.toast(f"Message padded to target {target_byte_size} bytes ({dummy_bit_len} bits)!")

def reset_embed_state(method_key_prefix):
    """Menghapus hasil yang disimpan di session_state saat input berubah."""
    image_key = f"{method_key_prefix}_stego_image_bytes"
    params_key = f"{method_key_prefix}_params_json"
    
    if image_key in st.session_state:
        st.session_state[image_key] = None
    if params_key in st.session_state:
        st.session_state[params_key] = None