# app.py

import streamlit as st
from helpers.message_generator import create_pattern_padded_message

# Impor konstanta
from constants import (
    METHOD_LSB, METHOD_PVD, METHOD_DCT, METHODS_ALL,
    DEFAULT_MESSAGE, DEFAULT_TARGET_BIT_SIZE
)

# Impor parameter default (diperlukan untuk inisialisasi state)
from methods.spatial.lsb import LSB_DEFAULT_PARAM
# Asumsikan file ini ada
# from methods.frequency.dct import DCT_DEFAULT_PARAM 
# from methods.hybrid.dct_lsb import HYBRID_DEFAULT_PARAM

# Impor fungsi UI
from ui_flows.lsb_ui import draw_lsb_embed_tab, draw_lsb_extract_tab
from ui_flows.pvd_ui import draw_pvd_embed_tab, draw_pvd_extract_tab
from ui_flows.dct_ui import draw_dct_embed_tab, draw_dct_extract_tab


# # --- Session State Initialization ---
# default_byte_size = (DEFAULT_TARGET_BIT_SIZE + 7) // 8
# default_padded_msg = create_pattern_padded_message(DEFAULT_MESSAGE, default_byte_size)

# # Pesan Embed
# if 'lsb_embed_msg' not in st.session_state: st.session_state.lsb_embed_msg = default_padded_msg
# if 'pvd_embed_msg' not in st.session_state: st.session_state.pvd_embed_msg = default_padded_msg
# if 'dct_embed_msg' not in st.session_state: st.session_state.dct_embed_msg = default_padded_msg

# Hasil Embed
if 'lsb_stego_image_bytes' not in st.session_state: st.session_state.lsb_stego_image_bytes = None
if 'lsb_params_json' not in st.session_state: st.session_state.lsb_params_json = None
if 'pvd_stego_image_bytes' not in st.session_state: st.session_state.pvd_stego_image_bytes = None
if 'pvd_params_json' not in st.session_state: st.session_state.pvd_params_json = None
if 'dct_stego_image_bytes' not in st.session_state: st.session_state.dct_stego_image_bytes = None
if 'dct_params_json' not in st.session_state: st.session_state.dct_params_json = None

# Parameter Ekstraksi
if 'lsb_extract_bits_per_channel' not in st.session_state:
    st.session_state.lsb_extract_bits_per_channel = LSB_DEFAULT_PARAM['bits_per_channel']
if 'lsb_extract_bit_length' not in st.session_state:
    st.session_state.lsb_extract_bit_length = 512
if 'pvd_extract_bit_length' not in st.session_state:
    st.session_state.pvd_extract_bit_length = 512
# (Tambahkan inisialisasi state untuk DCT dan Hybrid di sini)


# --- Sidebar Navigation ---
st.sidebar.title("Hybrid Steganography 🛡️")
st.sidebar.header("Spatial Domain")
st.sidebar.write(f"• {METHOD_LSB}")
st.sidebar.write(f"• {METHOD_PVD}")
st.sidebar.divider()
st.sidebar.header("Frequency Domain")
st.sidebar.write(f"• {METHOD_DCT}")
st.sidebar.divider()
st.sidebar.header("Hybrid")
st.sidebar.divider()
selected_method = st.sidebar.radio("Choose Method", METHODS_ALL, key="final_method_selector", label_visibility="collapsed")

# --- Konten Utama (Routing) ---
tab_embed, tab_extract = st.tabs(["Embed Message 🖼️", "Extract Message 🕵️"])

# --- Tab Embed ---
with tab_embed:
    st.title(f"Embed using {selected_method}")
    
    if selected_method == METHOD_LSB:
        draw_lsb_embed_tab()
        
    elif selected_method == METHOD_PVD:
        draw_pvd_embed_tab()
        
    elif selected_method == METHOD_DCT:
        draw_dct_embed_tab()

# --- Tab Extract ---
with tab_extract:
    st.title(f"Extract using {selected_method}")
    
    if selected_method == METHOD_LSB:
        draw_lsb_extract_tab()
        
    elif selected_method == METHOD_PVD:
        draw_pvd_extract_tab()
        
    elif selected_method == METHOD_DCT:
        draw_dct_extract_tab()