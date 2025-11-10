import streamlit as st
from helpers.message_generator import create_pattern_padded_message

from constants import (
    METHODS_ALL, DEFAULT_MESSAGE, DEFAULT_TARGET_BIT_SIZE,
    METHOD_LSB, METHOD_PVD, METHOD_EMD, 
    METHOD_DCT, METHOD_DWT, METHOD_FFT, 
    METHOD_DCT_LSB, METHOD_DCT_PVD, METHOD_DCT_EMD, 
    METHOD_DWT_LSB, METHOD_DWT_PVD, METHOD_DWT_EMD,
    METHOD_FFT_LSB, METHOD_FFT_PVD, METHOD_FFT_EMD # <--- DIPERBARUI
)

# Impor parameter default
from methods.spatial.lsb import LSB_DEFAULT_PARAM
from methods.spatial.pvd import PVD_DEFAULT_PARAM 
from methods.spatial.emd import EMD_DEFAULT_PARAM 
from methods.frequency.dct import DCT_DEFAULT_PARAM
from methods.frequency.dwt import DWT_DEFAULT_PARAM
from methods.frequency.fft import FFT_DEFAULT_PARAM
from methods.hybrid.dct_lsb import DCT_LSB_DEFAULT_PARAM
from methods.hybrid.dct_pvd import DCT_PVD_DEFAULT_PARAM
from methods.hybrid.dct_emd import DCT_EMD_DEFAULT_PARAM
from methods.hybrid.dwt_lsb import DWT_LSB_DEFAULT_PARAM
from methods.hybrid.dwt_pvd import DWT_PVD_DEFAULT_PARAM
from methods.hybrid.dwt_emd import DWT_EMD_DEFAULT_PARAM
from methods.hybrid.fft_lsb import FFT_LSB_DEFAULT_PARAM
from methods.hybrid.fft_pvd import FFT_PVD_DEFAULT_PARAM
from methods.hybrid.fft_emd import FFT_EMD_DEFAULT_PARAM # <--- DITAMBAHKAN

# Impor fungsi UI
from ui_flows.lsb_ui import draw_lsb_embed_tab, draw_lsb_extract_tab
from ui_flows.pvd_ui import draw_pvd_embed_tab, draw_pvd_extract_tab
from ui_flows.emd_ui import draw_emd_embed_tab, draw_emd_extract_tab
from ui_flows.dct_ui import draw_dct_embed_tab, draw_dct_extract_tab
from ui_flows.dwt_ui import draw_dwt_embed_tab, draw_dwt_extract_tab
from ui_flows.fft_ui import draw_fft_embed_tab, draw_fft_extract_tab
from ui_flows.dct_lsb_ui import draw_dct_lsb_embed_tab, draw_dct_lsb_extract_tab
from ui_flows.dct_pvd_ui import draw_dct_pvd_embed_tab, draw_dct_pvd_extract_tab
from ui_flows.dct_emd_ui import draw_dct_emd_embed_tab, draw_dct_emd_extract_tab
from ui_flows.dwt_lsb_ui import draw_dwt_lsb_embed_tab, draw_dwt_lsb_extract_tab
from ui_flows.dwt_pvd_ui import draw_dwt_pvd_embed_tab, draw_dwt_pvd_extract_tab
from ui_flows.dwt_emd_ui import draw_dwt_emd_embed_tab, draw_dwt_emd_extract_tab
from ui_flows.fft_lsb_ui import draw_fft_lsb_embed_tab, draw_fft_lsb_extract_tab
from ui_flows.fft_pvd_ui import draw_fft_pvd_embed_tab, draw_fft_pvd_extract_tab
from ui_flows.fft_emd_ui import draw_fft_emd_embed_tab, draw_fft_emd_extract_tab # <--- DITAMBAHKAN


# --- Session State Initialization ---
default_byte_size = (DEFAULT_TARGET_BIT_SIZE + 7) // 8
default_padded_msg = create_pattern_padded_message(DEFAULT_MESSAGE, default_byte_size)

# Pesan Embed
if 'lsb_embed_msg' not in st.session_state: st.session_state.lsb_embed_msg = default_padded_msg
if 'pvd_embed_msg' not in st.session_state: st.session_state.pvd_embed_msg = default_padded_msg
if 'emd_embed_msg' not in st.session_state: st.session_state.emd_embed_msg = default_padded_msg
if 'dct_embed_msg' not in st.session_state: st.session_state.dct_embed_msg = default_padded_msg
if 'dwt_embed_msg' not in st.session_state: st.session_state.dwt_embed_msg = default_padded_msg
if 'fft_embed_msg' not in st.session_state: st.session_state.fft_embed_msg = default_padded_msg
if 'dct_lsb_embed_msg' not in st.session_state: st.session_state.dct_lsb_embed_msg = default_padded_msg
if 'dct_pvd_embed_msg' not in st.session_state: st.session_state.dct_pvd_embed_msg = default_padded_msg
if 'dct_emd_embed_msg' not in st.session_state: st.session_state.dct_emd_embed_msg = default_padded_msg
if 'dwt_lsb_embed_msg' not in st.session_state: st.session_state.dwt_lsb_embed_msg = default_padded_msg
if 'dwt_pvd_embed_msg' not in st.session_state: st.session_state.dwt_pvd_embed_msg = default_padded_msg
if 'dwt_emd_embed_msg' not in st.session_state: st.session_state.dwt_emd_embed_msg = default_padded_msg
if 'fft_lsb_embed_msg' not in st.session_state: st.session_state.fft_lsb_embed_msg = default_padded_msg
if 'fft_pvd_embed_msg' not in st.session_state: st.session_state.fft_pvd_embed_msg = default_padded_msg
if 'fft_emd_embed_msg' not in st.session_state: st.session_state.fft_emd_embed_msg = default_padded_msg # <--- DITAMBAHKAN

# Hasil Embed
if 'lsb_stego_image_bytes' not in st.session_state: st.session_state.lsb_stego_image_bytes = None
if 'lsb_params_json' not in st.session_state: st.session_state.lsb_params_json = None
if 'pvd_stego_image_bytes' not in st.session_state: st.session_state.pvd_stego_image_bytes = None
if 'pvd_params_json' not in st.session_state: st.session_state.pvd_params_json = None
if 'emd_stego_image_bytes' not in st.session_state: st.session_state.emd_stego_image_bytes = None
if 'emd_params_json' not in st.session_state: st.session_state.emd_params_json = None
if 'dct_stego_image_bytes' not in st.session_state: st.session_state.dct_stego_image_bytes = None
if 'dct_params_json' not in st.session_state: st.session_state.dct_params_json = None
if 'dwt_stego_image_bytes' not in st.session_state: st.session_state.dwt_stego_image_bytes = None
if 'dwt_params_json' not in st.session_state: st.session_state.dwt_params_json = None
if 'fft_stego_image_bytes' not in st.session_state: st.session_state.fft_stego_image_bytes = None
if 'fft_params_json' not in st.session_state: st.session_state.fft_params_json = None
if 'dct_lsb_stego_image_bytes' not in st.session_state: st.session_state.dct_lsb_stego_image_bytes = None
if 'dct_lsb_params_json' not in st.session_state: st.session_state.dct_lsb_params_json = None
if 'dct_pvd_stego_image_bytes' not in st.session_state: st.session_state.dct_pvd_stego_image_bytes = None
if 'dct_pvd_params_json' not in st.session_state: st.session_state.dct_pvd_params_json = None
if 'dct_emd_stego_image_bytes' not in st.session_state: st.session_state.dct_emd_stego_image_bytes = None # <--- DITAMBAHKAN
if 'dct_emd_params_json' not in st.session_state: st.session_state.dct_emd_params_json = None
if 'dwt_lsb_stego_image_bytes' not in st.session_state: st.session_state.dwt_lsb_stego_image_bytes = None
if 'dwt_lsb_params_json' not in st.session_state: st.session_state.dwt_lsb_params_json = None
if 'dwt_pvd_stego_image_bytes' not in st.session_state: st.session_state.dwt_pvd_stego_image_bytes = None
if 'dwt_pvd_params_json' not in st.session_state: st.session_state.dwt_pvd_params_json = None
if 'dwt_emd_stego_image_bytes' not in st.session_state: st.session_state.dwt_emd_stego_image_bytes = None # <--- DITAMBAHKAN
if 'dwt_emd_params_json' not in st.session_state: st.session_state.dwt_emd_params_json = None # <--- DITAMBAHKAN
if 'fft_lsb_stego_image_bytes' not in st.session_state: st.session_state.fft_lsb_stego_image_bytes = None # <--- DITAMBAHKAN
if 'fft_lsb_params_json' not in st.session_state: st.session_state.fft_lsb_params_json = None # <--- DITAMBAHKAN
if 'fft_pvd_stego_image_bytes' not in st.session_state: st.session_state.fft_pvd_stego_image_bytes = None # <--- DITAMBAHKAN
if 'fft_pvd_params_json' not in st.session_state: st.session_state.fft_pvd_params_json = None # <--- DITAMBAHKAN
if 'fft_emd_stego_image_bytes' not in st.session_state: st.session_state.fft_emd_stego_image_bytes = None # <--- DITAMBAHKAN
if 'fft_emd_params_json' not in st.session_state: st.session_state.fft_emd_params_json = None # <--- DITAMBAHKAN

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

if 'fft_extract_r_in' not in st.session_state:
    st.session_state.fft_extract_r_in = FFT_DEFAULT_PARAM['r_in']
if 'fft_extract_r_out' not in st.session_state:
    st.session_state.fft_extract_r_out = FFT_DEFAULT_PARAM['r_out']
if 'fft_extract_header_repeat' not in st.session_state:
    st.session_state.fft_extract_header_repeat = FFT_DEFAULT_PARAM['header_repeat']
if 'fft_extract_payload_repeat' not in st.session_state:
    st.session_state.fft_extract_payload_repeat = FFT_DEFAULT_PARAM['payload_repeat']
if 'fft_extract_header_channel' not in st.session_state:
    st.session_state.fft_extract_header_channel = FFT_DEFAULT_PARAM['header_channel']
if 'fft_extract_payload_channel' not in st.session_state:
    st.session_state.fft_extract_payload_channel = FFT_DEFAULT_PARAM['payload_channel']
if 'fft_extract_mag_min_boost' not in st.session_state:
    st.session_state.fft_extract_mag_min_boost = FFT_DEFAULT_PARAM['mag_min_boost']
if 'fft_extract_bit_length' not in st.session_state:
    st.session_state.fft_extract_bit_length = 512 

if 'dct_lsb_extract_dct_lsb_ratio' not in st.session_state:
    st.session_state.dct_lsb_extract_dct_lsb_ratio = DCT_LSB_DEFAULT_PARAM['dct_lsb_ratio'][0]
if 'dct_lsb_extract_quant_factor' not in st.session_state:
    st.session_state.dct_lsb_extract_quant_factor = DCT_LSB_DEFAULT_PARAM['dct_params']['quant_factor']
if 'dct_lsb_extract_embed_positions' not in st.session_state:
    st.session_state.dct_lsb_extract_embed_positions = "Mid Frequencies"
if 'dct_lsb_extract_bits_per_channel' not in st.session_state:
    st.session_state.dct_lsb_extract_bits_per_channel = DCT_LSB_DEFAULT_PARAM['lsb_params']['bits_per_channel']
if 'dct_lsb_extract_dct_bit_length' not in st.session_state:
    st.session_state.dct_lsb_extract_dct_bit_length = 256
if 'dct_lsb_extract_lsb_bit_length' not in st.session_state:
    st.session_state.dct_lsb_extract_lsb_bit_length = 256

if 'dct_pvd_extract_dct_pvd_ratio' not in st.session_state:
    st.session_state.dct_pvd_extract_dct_pvd_ratio = DCT_PVD_DEFAULT_PARAM['dct_pvd_ratio'][0]
if 'dct_pvd_extract_quant_factor' not in st.session_state:
    st.session_state.dct_pvd_extract_quant_factor = DCT_PVD_DEFAULT_PARAM['dct_params']['quant_factor']
if 'dct_pvd_extract_embed_positions' not in st.session_state:
    st.session_state.dct_pvd_extract_embed_positions = "Mid Frequencies"
if 'dct_pvd_extract_dct_bit_length' not in st.session_state:
    st.session_state.dct_pvd_extract_dct_bit_length = 256
if 'dct_pvd_extract_pvd_bit_length' not in st.session_state:
    st.session_state.dct_pvd_extract_pvd_bit_length = 256

if 'dct_emd_extract_dct_emd_ratio' not in st.session_state:
    st.session_state.dct_emd_extract_dct_emd_ratio = DCT_EMD_DEFAULT_PARAM['dct_emd_ratio'][0]
if 'dct_emd_extract_quant_factor' not in st.session_state:
    st.session_state.dct_emd_extract_quant_factor = DCT_EMD_DEFAULT_PARAM['dct_params']['quant_factor']
if 'dct_emd_extract_embed_positions' not in st.session_state:
    st.session_state.dct_emd_extract_embed_positions = "Mid Frequencies" # Asumsi dari default
if 'dct_emd_extract_n' not in st.session_state:
    st.session_state.dct_emd_extract_n = DCT_EMD_DEFAULT_PARAM['emd_params']['n']
if 'dct_emd_extract_dct_bit_length' not in st.session_state:
    st.session_state.dct_emd_extract_dct_bit_length = 256
if 'dct_emd_extract_emd_bit_length' not in st.session_state:
    st.session_state.dct_emd_extract_emd_bit_length = 256

if 'dwt_lsb_extract_dwt_lsb_ratio' not in st.session_state:
    st.session_state.dwt_lsb_extract_dwt_lsb_ratio = DWT_LSB_DEFAULT_PARAM['dwt_lsb_ratio'][0]
if 'dwt_lsb_extract_wavelet' not in st.session_state:
    st.session_state.dwt_lsb_extract_wavelet = DWT_LSB_DEFAULT_PARAM['dwt_params']['wavelet']
if 'dwt_lsb_extract_level' not in st.session_state:
    st.session_state.dwt_lsb_extract_level = DWT_LSB_DEFAULT_PARAM['dwt_params']['level']
if 'dwt_lsb_extract_band' not in st.session_state:
    st.session_state.dwt_lsb_extract_band = DWT_LSB_DEFAULT_PARAM['dwt_params']['band']
if 'dwt_lsb_extract_embed_level' not in st.session_state:
    st.session_state.dwt_lsb_extract_embed_level = DWT_LSB_DEFAULT_PARAM['dwt_params']['embed_level']
if 'dwt_lsb_extract_delta' not in st.session_state:
    st.session_state.dwt_lsb_extract_delta = DWT_LSB_DEFAULT_PARAM['dwt_params']['delta']
if 'dwt_lsb_extract_robust_mode' not in st.session_state:
    st.session_state.dwt_lsb_extract_robust_mode = DWT_LSB_DEFAULT_PARAM['dwt_params']['robust_mode']
if 'dwt_lsb_extract_bits_per_channel' not in st.session_state:
    st.session_state.dwt_lsb_extract_bits_per_channel = DWT_LSB_DEFAULT_PARAM['lsb_params']['bits_per_channel']
if 'dwt_lsb_extract_dwt_bit_length' not in st.session_state:
    st.session_state.dwt_lsb_extract_dwt_bit_length = 256
if 'dwt_lsb_extract_lsb_bit_length' not in st.session_state:
    st.session_state.dwt_lsb_extract_lsb_bit_length = 256

if 'dwt_pvd_extract_dwt_pvd_ratio' not in st.session_state:
    st.session_state.dwt_pvd_extract_dwt_pvd_ratio = DWT_PVD_DEFAULT_PARAM['dwt_pvd_ratio'][0]
if 'dwt_pvd_extract_wavelet' not in st.session_state:
    st.session_state.dwt_pvd_extract_wavelet = DWT_PVD_DEFAULT_PARAM['dwt_params']['wavelet']
if 'dwt_pvd_extract_level' not in st.session_state:
    st.session_state.dwt_pvd_extract_level = DWT_PVD_DEFAULT_PARAM['dwt_params']['level']
if 'dwt_pvd_extract_band' not in st.session_state:
    st.session_state.dwt_pvd_extract_band = DWT_PVD_DEFAULT_PARAM['dwt_params']['band']
if 'dwt_pvd_extract_embed_level' not in st.session_state:
    st.session_state.dwt_pvd_extract_embed_level = DWT_PVD_DEFAULT_PARAM['dwt_params']['embed_level']
if 'dwt_pvd_extract_delta' not in st.session_state:
    st.session_state.dwt_pvd_extract_delta = DWT_PVD_DEFAULT_PARAM['dwt_params']['delta']
if 'dwt_pvd_extract_robust_mode' not in st.session_state:
    st.session_state.dwt_pvd_extract_robust_mode = DWT_PVD_DEFAULT_PARAM['dwt_params']['robust_mode']
if 'dwt_pvd_extract_dwt_bit_length' not in st.session_state:
    st.session_state.dwt_pvd_extract_dwt_bit_length = 256
if 'dwt_pvd_extract_pvd_bit_length' not in st.session_state:
    st.session_state.dwt_pvd_extract_pvd_bit_length = 256

if 'dwt_emd_extract_dwt_emd_ratio' not in st.session_state:
    st.session_state.dwt_emd_extract_dwt_emd_ratio = DWT_EMD_DEFAULT_PARAM['dwt_emd_ratio'][0]
# DWT params
if 'dwt_emd_extract_wavelet' not in st.session_state:
    st.session_state.dwt_emd_extract_wavelet = DWT_EMD_DEFAULT_PARAM['dwt_params']['wavelet']
if 'dwt_emd_extract_level' not in st.session_state:
    st.session_state.dwt_emd_extract_level = DWT_EMD_DEFAULT_PARAM['dwt_params']['level']
if 'dwt_emd_extract_band' not in st.session_state:
    st.session_state.dwt_emd_extract_band = DWT_EMD_DEFAULT_PARAM['dwt_params']['band']
if 'dwt_emd_extract_embed_level' not in st.session_state:
    st.session_state.dwt_emd_extract_embed_level = DWT_EMD_DEFAULT_PARAM['dwt_params']['embed_level']
if 'dwt_emd_extract_delta' not in st.session_state:
    st.session_state.dwt_emd_extract_delta = DWT_EMD_DEFAULT_PARAM['dwt_params']['delta']
if 'dwt_emd_extract_robust_mode' not in st.session_state:
    st.session_state.dwt_emd_extract_robust_mode = DWT_EMD_DEFAULT_PARAM['dwt_params']['robust_mode']
# EMD params
if 'dwt_emd_extract_n' not in st.session_state:
    st.session_state.dwt_emd_extract_n = DWT_EMD_DEFAULT_PARAM['emd_params']['n']
# Bit lengths
if 'dwt_emd_extract_dwt_bit_length' not in st.session_state:
    st.session_state.dwt_emd_extract_dwt_bit_length = 256
if 'dwt_emd_extract_emd_bit_length' not in st.session_state:
    st.session_state.dwt_emd_extract_emd_bit_length = 256

if 'fft_lsb_extract_fft_lsb_ratio' not in st.session_state:
    st.session_state.fft_lsb_extract_fft_lsb_ratio = FFT_LSB_DEFAULT_PARAM['fft_lsb_ratio'][0]
# FFT params
if 'fft_lsb_extract_r_in' not in st.session_state:
    st.session_state.fft_lsb_extract_r_in = FFT_LSB_DEFAULT_PARAM['fft_params']['r_in']
if 'fft_lsb_extract_r_out' not in st.session_state:
    st.session_state.fft_lsb_extract_r_out = FFT_LSB_DEFAULT_PARAM['fft_params']['r_out']
if 'fft_lsb_extract_header_repeat' not in st.session_state:
    st.session_state.fft_lsb_extract_header_repeat = FFT_LSB_DEFAULT_PARAM['fft_params']['header_repeat']
if 'fft_lsb_extract_payload_repeat' not in st.session_state:
    st.session_state.fft_lsb_extract_payload_repeat = FFT_LSB_DEFAULT_PARAM['fft_params']['payload_repeat']
if 'fft_lsb_extract_header_channel' not in st.session_state:
    st.session_state.fft_lsb_extract_header_channel = FFT_LSB_DEFAULT_PARAM['fft_params']['header_channel']
if 'fft_lsb_extract_payload_channel' not in st.session_state:
    st.session_state.fft_lsb_extract_payload_channel = FFT_LSB_DEFAULT_PARAM['fft_params']['payload_channel']
if 'fft_lsb_extract_mag_min_boost' not in st.session_state:
    st.session_state.fft_lsb_extract_mag_min_boost = FFT_LSB_DEFAULT_PARAM['fft_params']['mag_min_boost']
# LSB params
if 'fft_lsb_extract_bits_per_channel' not in st.session_state:
    st.session_state.fft_lsb_extract_bits_per_channel = FFT_LSB_DEFAULT_PARAM['lsb_params']['bits_per_channel']
# Bit lengths
if 'fft_lsb_extract_fft_bit_length' not in st.session_state:
    st.session_state.fft_lsb_extract_fft_bit_length = 256
if 'fft_lsb_extract_lsb_bit_length' not in st.session_state:
    st.session_state.fft_lsb_extract_lsb_bit_length = 256

if 'fft_pvd_extract_fft_pvd_ratio' not in st.session_state:
    st.session_state.fft_pvd_extract_fft_pvd_ratio = FFT_PVD_DEFAULT_PARAM['fft_pvd_ratio'][0]
# FFT params
if 'fft_pvd_extract_r_in' not in st.session_state:
    st.session_state.fft_pvd_extract_r_in = FFT_PVD_DEFAULT_PARAM['fft_params']['r_in']
if 'fft_pvd_extract_r_out' not in st.session_state:
    st.session_state.fft_pvd_extract_r_out = FFT_PVD_DEFAULT_PARAM['fft_params']['r_out']
if 'fft_pvd_extract_header_repeat' not in st.session_state:
    st.session_state.fft_pvd_extract_header_repeat = FFT_PVD_DEFAULT_PARAM['fft_params']['header_repeat']
if 'fft_pvd_extract_payload_repeat' not in st.session_state:
    st.session_state.fft_pvd_extract_payload_repeat = FFT_PVD_DEFAULT_PARAM['fft_params']['payload_repeat']
if 'fft_pvd_extract_header_channel' not in st.session_state:
    st.session_state.fft_pvd_extract_header_channel = FFT_PVD_DEFAULT_PARAM['fft_params']['header_channel']
if 'fft_pvd_extract_payload_channel' not in st.session_state:
    st.session_state.fft_pvd_extract_payload_channel = FFT_PVD_DEFAULT_PARAM['fft_params']['payload_channel']
if 'fft_pvd_extract_mag_min_boost' not in st.session_state:
    st.session_state.fft_pvd_extract_mag_min_boost = FFT_PVD_DEFAULT_PARAM['fft_params']['mag_min_boost']
if 'fft_pvd_extract_fft_bit_length' not in st.session_state:
    st.session_state.fft_pvd_extract_fft_bit_length = 256
if 'fft_pvd_extract_pvd_bit_length' not in st.session_state:
    st.session_state.fft_pvd_extract_pvd_bit_length = 256

if 'fft_emd_extract_fft_emd_ratio' not in st.session_state:
    st.session_state.fft_emd_extract_fft_emd_ratio = FFT_EMD_DEFAULT_PARAM['fft_emd_ratio'][0]
# FFT params
if 'fft_emd_extract_r_in' not in st.session_state:
    st.session_state.fft_emd_extract_r_in = FFT_EMD_DEFAULT_PARAM['fft_params']['r_in']
if 'fft_emd_extract_r_out' not in st.session_state:
    st.session_state.fft_emd_extract_r_out = FFT_EMD_DEFAULT_PARAM['fft_params']['r_out']
if 'fft_emd_extract_header_repeat' not in st.session_state:
    st.session_state.fft_emd_extract_header_repeat = FFT_EMD_DEFAULT_PARAM['fft_params']['header_repeat']
if 'fft_emd_extract_payload_repeat' not in st.session_state:
    st.session_state.fft_emd_extract_payload_repeat = FFT_EMD_DEFAULT_PARAM['fft_params']['payload_repeat']
if 'fft_emd_extract_header_channel' not in st.session_state:
    st.session_state.fft_emd_extract_header_channel = FFT_EMD_DEFAULT_PARAM['fft_params']['header_channel']
if 'fft_emd_extract_payload_channel' not in st.session_state:
    st.session_state.fft_emd_extract_payload_channel = FFT_EMD_DEFAULT_PARAM['fft_params']['payload_channel']
if 'fft_emd_extract_mag_min_boost' not in st.session_state:
    st.session_state.fft_emd_extract_mag_min_boost = FFT_EMD_DEFAULT_PARAM['fft_params']['mag_min_boost']
# EMD params
if 'fft_emd_extract_n' not in st.session_state:
    st.session_state.fft_emd_extract_n = FFT_EMD_DEFAULT_PARAM['emd_params']['n']
# Bit lengths
if 'fft_emd_extract_fft_bit_length' not in st.session_state:
    st.session_state.fft_emd_extract_fft_bit_length = 256
if 'fft_emd_extract_emd_bit_length' not in st.session_state:
    st.session_state.fft_emd_extract_emd_bit_length = 256
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
st.sidebar.write(f"• {METHOD_DWT}")
st.sidebar.write(f"• {METHOD_FFT}")
st.sidebar.divider()
st.sidebar.header("Hybrid")
st.sidebar.write(f"• {METHOD_DCT_LSB}")
st.sidebar.write(f"• {METHOD_DCT_PVD}")
st.sidebar.write(f"• {METHOD_DCT_EMD}")
st.sidebar.write(f"• {METHOD_DWT_LSB}")
st.sidebar.write(f"• {METHOD_DWT_PVD}")
st.sidebar.write(f"• {METHOD_FFT_LSB}")
st.sidebar.write(f"• {METHOD_FFT_EMD}") # <--- DITAMBAHKAN
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
    elif selected_method == METHOD_EMD:
        draw_emd_embed_tab()
    elif selected_method == METHOD_DCT:
        draw_dct_embed_tab()
    elif selected_method == METHOD_DWT:
        draw_dwt_embed_tab()
    elif selected_method == METHOD_FFT:
        draw_fft_embed_tab()
    elif selected_method == METHOD_DCT_LSB:
        draw_dct_lsb_embed_tab()
    elif selected_method == METHOD_DCT_PVD:
        draw_dct_pvd_embed_tab()
    elif selected_method == METHOD_DCT_EMD:
        draw_dct_emd_embed_tab()
    elif selected_method == METHOD_DWT_LSB:
        draw_dwt_lsb_embed_tab()
    elif selected_method == METHOD_DWT_PVD:
        draw_dwt_pvd_embed_tab()
    elif selected_method == METHOD_DWT_EMD:
        draw_dwt_emd_embed_tab()
    elif selected_method == METHOD_FFT_LSB:
        draw_fft_lsb_embed_tab()
    elif selected_method == METHOD_FFT_PVD:
        draw_fft_pvd_embed_tab()
    elif selected_method == METHOD_FFT_EMD: # <--- DITAMBAHKAN
        draw_fft_emd_embed_tab()

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
    elif selected_method == METHOD_DWT:
        draw_dwt_extract_tab()
    elif selected_method == METHOD_FFT:
        draw_fft_extract_tab()
    elif selected_method == METHOD_DCT_LSB:
        draw_dct_lsb_extract_tab()
    elif selected_method == METHOD_DCT_PVD:
        draw_dct_pvd_extract_tab()
    elif selected_method == METHOD_DCT_EMD:
        draw_dct_emd_extract_tab()
    elif selected_method == METHOD_DWT_LSB:
        draw_dwt_lsb_extract_tab()
    elif selected_method == METHOD_DWT_PVD:
        draw_dwt_pvd_extract_tab()
    elif selected_method == METHOD_DWT_EMD:
        draw_dwt_emd_extract_tab()
    elif selected_method == METHOD_FFT_LSB:
        draw_fft_lsb_extract_tab()
    elif selected_method == METHOD_FFT_PVD:
        draw_fft_pvd_extract_tab()
    elif selected_method == METHOD_FFT_EMD: # <--- DITAMBAHKAN
        draw_fft_emd_extract_tab()