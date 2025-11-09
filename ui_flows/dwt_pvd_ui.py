# ui_flows/dwt_pvd_ui.py

import streamlit as st
import json
import numpy as np
import pandas as pd
import cv2 
import pywt
from PIL import Image
from io import BytesIO

# Impor Lokal
from constants import (
    METHOD_DWT_PVD, DEFAULT_TARGET_BIT_SIZE
)
from ui_flows.utils import generate_dummy_message_callback, reset_embed_state, make_image_grid
from methods.hybrid.dwt_pvd import DWTPVDHybrid, DWT_PVD_DEFAULT_PARAM
# Kelas dasar diperlukan untuk RobustnessTester
from methods.frequency.dwt import DWTSteganography
from methods.spatial.pvd import PVDSteganography

from metrics.impercability import SteganographyMetrics
from metrics.robustness import RobustnessTester, ATTACK_CONFIGURATIONS
from helpers.message_binary import message_to_binary

# Daftar Wavelet yang Umum
WAVELET_LIST = ['haar', 'db1', 'db2', 'db3', 'bior1.1', 'bior2.2', 'bior3.3', 'sym2', 'sym3']
BAND_LIST = ['LL', 'LH', 'HL', 'HH']

def draw_dwt_pvd_embed_tab():
    """Menampilkan UI untuk tab Embed Hybrid DWT-PVD."""
    
    default_dwt_params = DWT_PVD_DEFAULT_PARAM['dwt_params']
    # pvd_params kosong
    default_ratio = DWT_PVD_DEFAULT_PARAM['dwt_pvd_ratio'][0]

    with st.expander("Configure Parameters"):
        hybrid_dwt_ratio = st.slider(
            "DWT Ratio", min_value=0.0, max_value=1.0, 
            value=default_ratio, step=0.1, 
            key="dwt_pvd_embed_ratio",
            on_change=reset_embed_state, args=("dwt_pvd",)
        )
        st.write(f"**DWT Ratio: {hybrid_dwt_ratio*100:.0f}%** | **PVD Ratio: {(1-hybrid_dwt_ratio)*100:.0f}%**")
        
        with st.container(border=True):
            st.write("**DWT Parameters**")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                hybrid_wavelet = st.selectbox(
                    "Wavelet Type", WAVELET_LIST, 
                    index=WAVELET_LIST.index(default_dwt_params['wavelet']),
                    key="dwt_pvd_embed_wavelet",
                    on_change=reset_embed_state, args=("dwt_pvd",)
                )
                hybrid_level = st.number_input(
                    "Decomposition Level (L)", min_value=1, max_value=8, 
                    value=default_dwt_params['level'],
                    key="dwt_pvd_embed_level",
                    on_change=reset_embed_state, args=("dwt_pvd",)
                )
                hybrid_band = st.selectbox(
                    "Sub-band", BAND_LIST, 
                    index=BAND_LIST.index(default_dwt_params['band']),
                    key="dwt_pvd_embed_band",
                    on_change=reset_embed_state, args=("dwt_pvd",)
                )
            with p_col2:
                hybrid_embed_level = st.number_input(
                    "Embedding Level (EL)", min_value=1, max_value=int(st.session_state.dwt_pvd_embed_level), 
                    value=min(default_dwt_params['embed_level'], st.session_state.dwt_pvd_embed_level),
                    key="dwt_pvd_embed_embed_level",
                    on_change=reset_embed_state, args=("dwt_pvd",),
                    help="Level dekomposisi untuk penyisipan (1 <= EL <= L)"
                )
                hybrid_delta = st.number_input(
                    "Quantization Step (Delta)", min_value=1.0, max_value=100.0, 
                    value=default_dwt_params['delta'], 
                    key="dwt_pvd_embed_delta",
                    on_change=reset_embed_state, args=("dwt_pvd",)
                )
                hybrid_robust_mode = st.checkbox(
                    "Enable Robust Mode (DWT)", 
                    value=default_dwt_params['robust_mode'], 
                    key="dwt_pvd_embed_robust_mode",
                    on_change=reset_embed_state, args=("dwt_pvd",)
                )

        with st.container(border=True):
            st.write("**PVD Parameters**")
            st.info("PVD does not have configurable parameters.")

    st.subheader("Message Input")
    col1, col2 = st.columns([0.6, 0.4]) 
    with col1:
        hybrid_embed_msg = st.text_area("Your Secret Message", height=150, key="dwt_pvd_embed_msg", label_visibility="collapsed",
                                        on_change=reset_embed_state, args=("dwt_pvd",))
    with col2:
        with st.container(border=True):
            st.write("Dummy Message Helper")
            dummy_bit_len = st.number_input("Target Bit Length", min_value=8, value=DEFAULT_TARGET_BIT_SIZE, step=8, key="dwt_pvd_dummy_bit_len")
            st.button("Generate Dummy Message", key="dwt_pvd_dummy_btn", on_click=generate_dummy_message_callback, args=("dwt_pvd_dummy_bit_len", "dwt_pvd_embed_msg"))

    st.subheader("Cover Image")
    cover_image_file = st.file_uploader("Upload Cover Image", type=["png", "jpg", "bmp"], key="dwt_pvd_embed_img", label_visibility="collapsed",
                                        on_change=reset_embed_state, args=("dwt_pvd",))
        
    if cover_image_file is not None:
        st.image(cover_image_file, caption="Uploaded Cover Image", use_container_width=True)
        
    st.divider()
    embed_button = st.button("Embed Message", type="primary", key="dwt_pvd_embed_btn")

    if embed_button:
        if cover_image_file is None:
            st.error("Please upload a cover image first!")
            st.session_state.dwt_pvd_stego_image_bytes = None
            st.session_state.dwt_pvd_params_json = None
        else:
            with st.spinner("Embedding message, please wait... ⏳"):
                try:
                    file_bytes = np.asarray(bytearray(cover_image_file.read()), dtype=np.uint8)
                    cover_image_rgb = cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                    
                    dwt_params_to_use = {
                        'wavelet': hybrid_wavelet,
                        'level': hybrid_level,
                        'band': hybrid_band,
                        'embed_level': hybrid_embed_level,
                        'delta': hybrid_delta,
                        'robust_mode': hybrid_robust_mode
                    }
                    pvd_params_to_use = {}
                    
                    hybrid_stego = DWTPVDHybrid(
                        dwt_pvd_ratio=(hybrid_dwt_ratio, 1.0 - hybrid_dwt_ratio),
                        dwt_params=dwt_params_to_use,
                        pvd_params=pvd_params_to_use
                    )
                    
                    stego_image_np, (dwt_len, pvd_len) = hybrid_stego.embed(cover_image_rgb, hybrid_embed_msg)

                    stego_image_pil = Image.fromarray(stego_image_np)
                    buffer = BytesIO()
                    stego_image_pil.save(buffer, format="PNG")
                    
                    parameters_to_save = {
                        "method": METHOD_DWT_PVD,
                        "dwt_pvd_ratio": (hybrid_dwt_ratio, 1.0 - hybrid_dwt_ratio),
                        "dwt_params": dwt_params_to_use,
                        "pvd_params": pvd_params_to_use,
                        "message_bit_lengths": (dwt_len, pvd_len)
                    }
                    
                    st.session_state.dwt_pvd_stego_image_bytes = buffer.getvalue()
                    st.session_state.dwt_pvd_params_json = json.dumps(parameters_to_save, indent=4, default=tuple)
                            
                except ValueError as e:
                    st.error(f"Embedding Failed: {e}")
                    st.session_state.dwt_pvd_stego_image_bytes = None
                    st.session_state.dwt_pvd_params_json = None
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    st.session_state.dwt_pvd_stego_image_bytes = None
                    st.session_state.dwt_pvd_params_json = None
    
    if st.session_state.dwt_pvd_stego_image_bytes is not None:
        st.subheader("Embedding Results")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.write("**Stego-Image**")
            st.image(st.session_state.dwt_pvd_stego_image_bytes, caption="Steganographic Image", use_container_width=True)
            st.download_button(label="Download Image", data=st.session_state.dwt_pvd_stego_image_bytes, file_name="dwt_pvd_stego_image.png", mime="image/png")
        with res_col2:
            st.write("**Parameters Used**")
            st.download_button(label="Download Parameters", data=st.session_state.dwt_pvd_params_json, file_name="dwt_pvd_parameters.json", mime="application/json")


def draw_dwt_pvd_extract_tab():
    """Menampilkan UI untuk tab Extract Hybrid DWT-PVD."""
    
    st.subheader("Load Parameters (Optional)")
    param_file = st.file_uploader(
        "Upload parameters.json file", 
        type=["json"], 
        key="dwt_pvd_extract_param_file"
    )
    
    if param_file is not None:
        try:
            param_data = json.loads(param_file.read().decode('utf-8'))
            
            if param_data.get('method') != METHOD_DWT_PVD:
                st.warning(f"File parameter ini untuk '{param_data.get('method')}', tetapi Anda memilih Hybrid.")
            
            if param_data.get('dwt_pvd_ratio') is not None:
                st.session_state.dwt_pvd_extract_dwt_pvd_ratio = param_data['dwt_pvd_ratio'][0]
            
            if param_data.get('dwt_params') is not None:
                dwt_p = param_data['dwt_params']
                if dwt_p.get('wavelet') is not None: st.session_state.dwt_pvd_extract_wavelet = dwt_p['wavelet']
                if dwt_p.get('level') is not None: st.session_state.dwt_pvd_extract_level = dwt_p['level']
                if dwt_p.get('band') is not None: st.session_state.dwt_pvd_extract_band = dwt_p['band']
                if dwt_p.get('embed_level') is not None: st.session_state.dwt_pvd_extract_embed_level = dwt_p['embed_level']
                if dwt_p.get('delta') is not None: st.session_state.dwt_pvd_extract_delta = dwt_p['delta']
                if dwt_p.get('robust_mode') is not None: st.session_state.dwt_pvd_extract_robust_mode = dwt_p['robust_mode']

            # pvd_params kosong
            
            if param_data.get('message_bit_lengths') is not None:
                st.session_state.dwt_pvd_extract_dwt_bit_length = param_data['message_bit_lengths'][0]
                st.session_state.dwt_pvd_extract_pvd_bit_length = param_data['message_bit_lengths'][1]

            st.toast("Parameters loaded successfully!")

        except Exception as e:
            st.error(f"Failed to load parameters: {e}")

    with st.expander("Configure Parameters"):
        hybrid_extract_ratio = st.slider(
            "DWT Ratio", min_value=0.0, max_value=1.0, 
            step=0.1, 
            key="dwt_pvd_extract_dwt_pvd_ratio"
        )
        st.write(f"**DWT Ratio: {hybrid_extract_ratio*100:.0f}%** | **PVD Ratio: {(1-hybrid_extract_ratio)*100:.0f}%**")
        
        with st.container(border=True):
            st.write("**DWT Parameters**")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                hybrid_extract_wavelet = st.selectbox(
                    "Wavelet Type", WAVELET_LIST, 
                    key="dwt_pvd_extract_wavelet"
                )
                hybrid_extract_level = st.number_input(
                    "Decomposition Level (L)", min_value=1, max_value=8, 
                    key="dwt_pvd_extract_level"
                )
                hybrid_extract_band = st.selectbox(
                    "Sub-band", BAND_LIST, 
                    key="dwt_pvd_extract_band"
                )
            with p_col2:
                hybrid_extract_embed_level = st.number_input(
                    "Embedding Level (EL)", min_value=1, 
                    max_value=int(st.session_state.dwt_pvd_extract_level),
                    key="dwt_pvd_extract_embed_level"
                )
                hybrid_extract_delta = st.number_input(
                    "Quantization Step (Delta)", min_value=1.0, max_value=100.0, 
                    key="dwt_pvd_extract_delta"
                )
                hybrid_extract_robust_mode = st.checkbox(
                    "Enable Robust Mode (DWT)", 
                    key="dwt_pvd_extract_robust_mode"
                )
        
        with st.container(border=True):
            st.write("**PVD Parameters**")
            st.info("PVD does not have configurable parameters.")

    st.subheader("Upload Stego-Image")
    stego_image_file_extract = st.file_uploader(
        "Upload the image you want to extract from", 
        type=["png", "jpg", "bmp"], 
        key="dwt_pvd_extract_img",
        label_visibility="collapsed"
    )
    if stego_image_file_extract is not None:
        st.image(stego_image_file_extract, caption="Uploaded Stego-Image", use_container_width=True)

    st.subheader("Message Bit Lengths")
    col1, col2 = st.columns(2)
    with col1:
        dwt_bit_length_extract = st.number_input(
            "DWT Bit Length", min_value=1, 
            key="dwt_pvd_extract_dwt_bit_length"
        )
    with col2:
        pvd_bit_length_extract = st.number_input(
            "PVD Bit Length", min_value=1, 
            key="dwt_pvd_extract_pvd_bit_length"
        )
    
    st.subheader("Optional Inputs for Metrics")
    col1_opt, col2_opt = st.columns(2)
    with col1_opt:
        optional_cover_image = st.file_uploader(
            "Original Cover Image (for Imperceptibility)", 
            type=["png", "jpg", "bmp"], 
            key="dwt_pvd_extract_cover"
        )
    with col2_opt:
        optional_original_message = st.text_area(
            "Original Message (for Robustness)", 
            height=155, 
            key="dwt_pvd_extract_msg"
        )

    st.divider()
    extract_button = st.button("Extract Message", type="primary", key="dwt_pvd_extract_btn")
    
    if extract_button:
        if stego_image_file_extract is None:
            st.error("Please upload a stego-image first!")
        else:
            with st.spinner("Extracting message and calculating metrics... ⏳"):
                try:
                    file_bytes = np.asarray(bytearray(stego_image_file_extract.read()), dtype=np.uint8)
                    stego_image_rgb = cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                    
                    dwt_params_ext = {
                        'wavelet': hybrid_extract_wavelet,
                        'level': hybrid_extract_level,
                        'band': hybrid_extract_band,
                        'embed_level': hybrid_extract_embed_level,
                        'delta': hybrid_extract_delta,
                        'robust_mode': hybrid_extract_robust_mode
                    }
                    pvd_params_ext = {}
                    
                    hybrid_stego_extract = DWTPVDHybrid(
                        dwt_pvd_ratio=(hybrid_extract_ratio, 1.0 - hybrid_extract_ratio),
                        dwt_params=dwt_params_ext,
                        pvd_params=pvd_params_ext
                    )
                    
                    bit_lengths_tuple = (dwt_bit_length_extract, pvd_bit_length_extract)
                    
                    extracted_message = hybrid_stego_extract.extract(stego_image_rgb, bit_lengths_tuple)
                    
                    st.subheader("Extracted Message")
                    st.text_area("Result", value=extracted_message, height=100, disabled=True)
                    
                    st.divider()
                    st.subheader("Imperceptibility Metrics")
                    if optional_cover_image is not None:
                        with st.spinner("Calculating imperceptibility metrics..."):
                            cover_file_bytes = np.asarray(bytearray(optional_cover_image.read()), dtype=np.uint8)
                            cover_image_rgb = cv2.cvtColor(cv2.imdecode(cover_file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                            metrics = SteganographyMetrics(cover_image_rgb, stego_image_rgb)
                            all_metrics = metrics.get_all_metrics()
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            metric_col1.metric(label="MSE", value=f"{all_metrics['mse']:.4e}")
                            metric_col2.metric(label="PSNR (dB)", value=f"{all_metrics['psnr']:.4f}")
                            metric_col3.metric(label="SSIM", value=f"{all_metrics['ssim']:.8f}")
                    else:
                        st.info("Upload the original cover image to calculate imperceptibility metrics.")
                    
                    st.divider()
                    st.subheader("Robustness Test (Bit Error Ratio - BER)")
                    if optional_original_message:
                        with st.spinner("Calculating robustness (BER) against 32 attacks..."):
                            original_binary_message = message_to_binary(optional_original_message)
                            
                            tester = RobustnessTester(hybrid_stego_extract, original_binary_message)
                            
                            results = tester.run_all_tests(
                                stego_image=stego_image_rgb,
                                attack_configurations=ATTACK_CONFIGURATIONS,
                                bit_lengths=bit_lengths_tuple
                            )
                            ber_data = []
                            image_results = []
                            for attack_label, (ber, attacked_img) in results.items():
                                ber_data.append({'Attack': attack_label, 'BER': ber})
                                if attacked_img is not None:
                                    image_results.append((attack_label, ber, attacked_img))
                            
                            df_ber = pd.DataFrame(ber_data)
                            st.dataframe(df_ber, height=300)

                            st.subheader("Attacked Images")
                            if image_results:
                                make_image_grid(image_results, num_columns=4)
                            
                    else:
                        st.info("Enter the original message to calculate robustness (BER).")

                except Exception as e:
                    st.error(f"Extraction or Metrics Failed: {e}")