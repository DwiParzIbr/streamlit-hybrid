# app.py

import streamlit as st
from helpers.message_generator import create_pattern_padded_message

# Impor konstanta
from constants import (
    METHOD_LSB, METHOD_PVD, METHOD_EMD, METHOD_DCT, METHOD_DWT, METHODS_ALL, # <--- DIPERBARUI
    DEFAULT_MESSAGE, DEFAULT_TARGET_BIT_SIZE
)

# Impor parameter default
from methods.spatial.lsb import LSB_DEFAULT_PARAM
from methods.spatial.pvd import PVD_DEFAULT_PARAM
from methods.spatial.emd import EMD_DEFAULT_PARAM
from methods.frequency.dct import DCT_DEFAULT_PARAM
from methods.frequency.dwt import DWT_DEFAULT_PARAM # <--- DITAMBAHKAN

# Impor fungsi UI
from ui_flows.lsb_ui import draw_lsb_embed_tab, draw_lsb_extract_tab
from ui_flows.pvd_ui import draw_pvd_embed_tab, draw_pvd_extract_tab
from ui_flows.emd_ui import draw_emd_embed_tab, draw_emd_extract_tab
from ui_flows.dct_ui import draw_dct_embed_tab, draw_dct_extract_tab
from ui_flows.dwt_ui import draw_dwt_embed_tab, draw_dwt_extract_tab # <--- DITAMBAHKAN


# --- Session State Initialization ---
default_byte_size = (DEFAULT_TARGET_BIT_SIZE + 7) // 8
default_padded_msg = create_pattern_padded_message(DEFAULT_MESSAGE, default_byte_size)

# Pesan Embed
if 'lsb_embed_msg' not in st.session_state: st.session_state.lsb_embed_msg = default_padded_msg
if 'pvd_embed_msg' not in st.session_state: st.session_state.pvd_embed_msg = default_padded_msg
if 'emd_embed_msg' not in st.session_state: st.session_state.emd_embed_msg = default_padded_msg
if 'dct_embed_msg' not in st.session_state: st.session_state.dct_embed_msg = default_padded_msg
if 'dwt_embed_msg' not in st.session_state: st.session_state.dwt_embed_msg = default_padded_msg # <--- DITAMBAHKAN

# Hasil Embed
if 'lsb_stego_image_bytes' not in st.session_state: st.session_state.lsb_stego_image_bytes = None
if 'lsb_params_json' not in st.session_state: st.session_state.lsb_params_json = None
if 'pvd_stego_image_bytes' not in st.session_state: st.session_state.pvd_stego_image_bytes = None
if 'pvd_params_json' not in st.session_state: st.session_state.pvd_params_json = None
if 'emd_stego_image_bytes' not in st.session_state: st.session_state.emd_stego_image_bytes = None
if 'emd_params_json' not in st.session_state: st.session_state.emd_params_json = None
if 'dct_stego_image_bytes' not in st.session_state: st.session_state.dct_stego_image_bytes = None
if 'dct_params_json' not in st.session_state: st.session_state.dct_params_json = None
if 'dwt_stego_image_bytes' not in st.session_state: st.session_state.dwt_stego_image_bytes = None # <--- DITAMBAHKAN
if 'dwt_params_json' not in st.session_state: st.session_state.dwt_params_json = None # <--- DITAMBAHKAN

# Parameter Ekstraksi
if 'lsb_extract_bits_per_channel' not in st.session_state:
    st.session_state.lsb_extract_bits_per_channel = LSB_DEFAULT_PARAM['bits_per_channel']
if 'lsb_extract_bit_length' not in st.session_state:
    st.session_state.lsb_extract_bit_length = 512
if 'pvd_extract_bit_length' not in st.session_state:
    st.session_state.pvd_extract_bit_length = 512
if 'emd_extract_n' not in st.session_state:
    st.session_state.emd_extract_n = EMD_DEFAULT_PARAM['n']
if 'emd_extract_bit_length' not in st.session_state:
    st.session_state.emd_extract_bit_length = 512
if 'dct_extract_block_size' not in st.session_state:
    st.session_state.dct_extract_block_size = DCT_DEFAULT_PARAM['block_size']
if 'dct_extract_quant_factor' not in st.session_state:
    st.session_state.dct_extract_quant_factor = DCT_DEFAULT_PARAM['quant_factor']
if 'dct_extract_embed_positions' not in st.session_state:
    st.session_state.dct_extract_embed_positions = "Mid Frequencies" 
if 'dct_extract_bit_length' not in st.session_state:
    st.session_state.dct_extract_bit_length = 512

# --- PERUBAHAN DI SINI: Inisialisasi state DWT ---
if 'dwt_extract_wavelet' not in st.session_state:
    st.session_state.dwt_extract_wavelet = DWT_DEFAULT_PARAM['wavelet']
if 'dwt_extract_level' not in st.session_state:
    st.session_state.dwt_extract_level = DWT_DEFAULT_PARAM['level']
if 'dwt_extract_band' not in st.session_state:
    st.session_state.dwt_extract_band = DWT_DEFAULT_PARAM['band']
if 'dwt_extract_embed_level' not in st.session_state:
    st.session_state.dwt_extract_embed_level = DWT_DEFAULT_PARAM['embed_level']
if 'dwt_extract_delta' not in st.session_state:
    st.session_state.dwt_extract_delta = DWT_DEFAULT_PARAM['delta']
if 'dwt_extract_robust_mode' not in st.session_state:
    st.session_state.dwt_extract_robust_mode = DWT_DEFAULT_PARAM['robust_mode']
if 'dwt_extract_bit_length' not in st.session_state:
    st.session_state.dwt_extract_bit_length = 512
# --- AKHIR PERUBAHAN ---


# --- Sidebar Navigation ---
st.sidebar.title("Hybrid Steganography 🛡️")
st.sidebar.header("Spatial Domain")
st.sidebar.write(f"• {METHOD_LSB}")
st.sidebar.write(f"• {METHOD_PVD}")
st.sidebar.write(f"• {METHOD_EMD}")
st.sidebar.divider()
st.sidebar.header("Frequency Domain")
st.sidebar.write(f"• {METHOD_DCT}")
st.sidebar.write(f"• {METHOD_DWT}") # <--- DITAMBAHKAN
st.sidebar.divider()
# st.sidebar.header("Hybrid") # Dihapus
# st.sidebar.divider()
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
    elif selected_method == METHOD_EMD:
        draw_emd_embed_tab()
    elif selected_method == METHOD_DCT:
        draw_dct_embed_tab()
    elif selected_method == METHOD_DWT: # <--- DITAMBAHKAN
        draw_dwt_embed_tab()

# --- Tab Extract ---
with tab_extract:
    st.title(f"Extract using {selected_method}")
    
    if selected_method == METHOD_LSB:
        draw_lsb_extract_tab()
    elif selected_method == METHOD_PVD:
        draw_pvd_extract_tab()
    elif selected_method == METHOD_EMD:
        draw_emd_extract_tab()
    elif selected_method == METHOD_DCT:
        draw_dct_extract_tab()
    elif selected_method == METHOD_DWT: # <--- DITAMBAHKAN
        draw_dwt_extract_tab()