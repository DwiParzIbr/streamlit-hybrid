# ui_flows/dwt_ui.py

import streamlit as st
import json
import numpy as np
import pandas as pd
import cv2 
import pywt # Diperlukan untuk daftar wavelet
from PIL import Image
from io import BytesIO

# Impor Lokal
from constants import METHOD_DWT, DEFAULT_TARGET_BIT_SIZE
from ui_flows.utils import generate_dummy_message_callback, reset_embed_state, make_image_grid
from methods.frequency.dwt import DWTSteganography, DWT_DEFAULT_PARAM
from metrics.impercability import SteganographyMetrics
from metrics.robustness import RobustnessTester, ATTACK_CONFIGURATIONS
from helpers.message_binary import message_to_binary

# Daftar Wavelet yang Umum
WAVELET_LIST = ['haar', 'db1', 'db2', 'db3', 'bior1.1', 'bior2.2', 'bior3.3', 'sym2', 'sym3']
BAND_LIST = ['LL', 'LH', 'HL', 'HH']

def draw_dwt_embed_tab():
    """Menampilkan UI untuk tab Embed DWT."""
    
    with st.expander("Configure Parameters"):
        # Pisahkan parameter menjadi dua kolom agar tidak terlalu panjang
        p_col1, p_col2 = st.columns(2)
        
        with p_col1:
            dwt_wavelet = st.selectbox(
                "Wavelet Type", WAVELET_LIST, 
                index=WAVELET_LIST.index(DWT_DEFAULT_PARAM['wavelet']),
                key="dwt_embed_wavelet",
                on_change=reset_embed_state, args=("dwt",)
            )
            dwt_level = st.number_input(
                "Decomposition Level (L)", min_value=1, max_value=8, 
                value=DWT_DEFAULT_PARAM['level'],
                key="dwt_embed_level",
                on_change=reset_embed_state, args=("dwt",)
            )
            dwt_band = st.selectbox(
                "Sub-band", BAND_LIST, 
                index=BAND_LIST.index(DWT_DEFAULT_PARAM['band']),
                key="dwt_embed_band",
                on_change=reset_embed_state, args=("dwt",)
            )
        
        with p_col2:
            dwt_embed_level = st.number_input(
                "Embedding Level (EL)", min_value=1, max_value=int(st.session_state.dwt_embed_level), 
                value=min(DWT_DEFAULT_PARAM['embed_level'], st.session_state.dwt_embed_level),
                key="dwt_embed_embed_level",
                on_change=reset_embed_state, args=("dwt",),
                help="Level dekomposisi untuk penyisipan (1 <= EL <= L)"
            )
            dwt_delta = st.number_input(
                "Quantization Step (Delta)", min_value=1.0, max_value=100.0, 
                value=DWT_DEFAULT_PARAM['delta'], 
                key="dwt_embed_delta",
                on_change=reset_embed_state, args=("dwt",)
            )
            dwt_robust_mode = st.checkbox(
                "Enable Robust Mode", 
                value=DWT_DEFAULT_PARAM['robust_mode'], 
                key="dwt_embed_robust_mode",
                on_change=reset_embed_state, args=("dwt",)
            )

    st.subheader("Message Input")
    col1, col2 = st.columns([0.6, 0.4]) 
    with col1:
        dwt_embed_msg = st.text_area("Your Secret Message", height=150, key="dwt_embed_msg", label_visibility="collapsed",
                                     on_change=reset_embed_state, args=("dwt",))
    with col2:
        with st.container(border=True):
            st.write("Dummy Message Helper")
            dummy_bit_len = st.number_input("Target Bit Length", min_value=8, value=DEFAULT_TARGET_BIT_SIZE, step=8, key="dwt_dummy_bit_len")
            st.button("Generate Dummy Message", key="dwt_dummy_btn", on_click=generate_dummy_message_callback, args=("dwt_dummy_bit_len", "dwt_embed_msg"))
    
    st.subheader("Cover Image")
    cover_image_file = st.file_uploader("Upload Cover Image", type=["png", "jpg", "bmp"], key="dwt_embed_img", label_visibility="collapsed",
                                        on_change=reset_embed_state, args=("dwt",))
        
    if cover_image_file is not None:
        st.image(cover_image_file, caption="Uploaded Cover Image", use_container_width=True)
        
    st.divider()
    embed_button = st.button("Embed Message", type="primary", key="dwt_embed_btn")

    if embed_button:
        if cover_image_file is None:
            st.error("Please upload a cover image first!")
            st.session_state.dwt_stego_image_bytes = None 
            st.session_state.dwt_params_json = None
        else:
            with st.spinner("Embedding message, please wait... ⏳"):
                try:
                    file_bytes = np.asarray(bytearray(cover_image_file.read()), dtype=np.uint8)
                    cover_image_rgb = cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                    
                    dwt_stego = DWTSteganography(
                        wavelet=dwt_wavelet,
                        level=dwt_level,
                        band=dwt_band,
                        delta=dwt_delta,
                        embed_level=dwt_embed_level,
                        robust_mode=dwt_robust_mode
                        # Parameter robust lainnya (reps, seed) menggunakan default
                    )
                    stego_image_np, final_bit_length = dwt_stego.embed(cover_image_rgb, dwt_embed_msg)

                    stego_image_pil = Image.fromarray(stego_image_np)
                    buffer = BytesIO()
                    stego_image_pil.save(buffer, format="PNG")
                    
                    parameters_to_save = {
                        "method": METHOD_DWT,
                        "wavelet": dwt_wavelet,
                        "level": dwt_level,
                        "band": dwt_band,
                        "embed_level": dwt_embed_level,
                        "delta": dwt_delta,
                        "robust_mode": dwt_robust_mode,
                        "message_bit_length": final_bit_length
                    }
                    
                    st.session_state.dwt_stego_image_bytes = buffer.getvalue()
                    st.session_state.dwt_params_json = json.dumps(parameters_to_save, indent=4)
                            
                except ValueError as e:
                    st.error(f"Embedding Failed: {e}")
                    st.session_state.dwt_stego_image_bytes = None
                    st.session_state.dwt_params_json = None
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    st.session_state.dwt_stego_image_bytes = None
                    st.session_state.dwt_params_json = None
    
    if st.session_state.dwt_stego_image_bytes is not None:
        st.subheader("Embedding Results")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.write("**Stego-Image**")
            st.image(st.session_state.dwt_stego_image_bytes, caption="Steganographic Image", use_container_width=True)
            st.download_button(label="Download Image", data=st.session_state.dwt_stego_image_bytes, file_name="dwt_stego_image.png", mime="image/png")
        with res_col2:
            st.write("**Parameters Used**")
            st.download_button(label="Download Parameters", data=st.session_state.dwt_params_json, file_name="dwt_parameters.json", mime="application/json")


def draw_dwt_extract_tab():
    """Menampilkan UI untuk tab Extract DWT."""
    
    st.subheader("Load Parameters (Optional)")
    param_file = st.file_uploader(
        "Upload parameters.json file", 
        type=["json"], 
        key="dwt_extract_param_file"
    )
    
    if param_file is not None:
        try:
            param_data = json.loads(param_file.read().decode('utf-8'))
            
            if param_data.get('method') != METHOD_DWT:
                st.warning(f"File parameter ini untuk '{param_data.get('method')}', tetapi Anda memilih DWT.")
            
            if param_data.get('wavelet') is not None:
                st.session_state.dwt_extract_wavelet = param_data['wavelet']
            if param_data.get('level') is not None:
                st.session_state.dwt_extract_level = param_data['level']
            if param_data.get('band') is not None:
                st.session_state.dwt_extract_band = param_data['band']
            if param_data.get('embed_level') is not None:
                st.session_state.dwt_extract_embed_level = param_data['embed_level']
            if param_data.get('delta') is not None:
                st.session_state.dwt_extract_delta = param_data['delta']
            if param_data.get('robust_mode') is not None:
                st.session_state.dwt_extract_robust_mode = param_data['robust_mode']
            if param_data.get('message_bit_length') is not None:
                st.session_state.dwt_extract_bit_length = param_data['message_bit_length']

            st.toast("Parameters loaded successfully!")

        except Exception as e:
            st.error(f"Failed to load parameters: {e}")

    with st.expander("Configure Parameters"):
        p_col1, p_col2 = st.columns(2)
        
        with p_col1:
            # --- PERBAIKAN DI SINI: 'index=' dihapus ---
            dwt_wavelet_extract = st.selectbox(
                "Wavelet Type", WAVELET_LIST, 
                key="dwt_extract_wavelet"
            )
            dwt_level_extract = st.number_input(
                "Decomposition Level (L)", min_value=1, max_value=8, 
                key="dwt_extract_level"
            )
            # --- PERBAIKAN DI SINI: 'index=' dihapus ---
            dwt_band_extract = st.selectbox(
                "Sub-band", BAND_LIST, 
                key="dwt_extract_band"
            )
        
        with p_col2:
            dwt_embed_level_extract = st.number_input(
                "Embedding Level (EL)", min_value=1, max_value=int(st.session_state.dwt_extract_level),
                key="dwt_extract_embed_level"
            )
            dwt_delta_extract = st.number_input(
                "Quantization Step (Delta)", min_value=1.0, max_value=100.0, 
                key="dwt_extract_delta"
            )
            dwt_robust_mode_extract = st.checkbox(
                "Enable Robust Mode", 
                key="dwt_extract_robust_mode"
            )

    st.subheader("Upload Stego-Image")
    stego_image_file_extract = st.file_uploader(
        "Upload the image you want to extract from", 
        type=["png", "jpg", "bmp"], 
        key="dwt_extract_img", 
        label_visibility="collapsed"
    )
    if stego_image_file_extract is not None:
        st.image(stego_image_file_extract, caption="Uploaded Stego-Image", use_container_width=True)

    st.subheader("Message Bit Length")
    message_bit_length_extract = st.number_input(
        "Enter the bit length of the secret message", 
        min_value=1, 
        key="dwt_extract_bit_length"
    )
    
    st.subheader("Optional Inputs for Metrics")
    col1, col2 = st.columns(2)
    with col1:
        optional_cover_image = st.file_uploader(
            "Original Cover Image (for Imperceptibility)", 
            type=["png", "jpg", "bmp"], 
            key="dwt_extract_cover"
        )
    with col2:
        optional_original_message = st.text_area(
            "Original Message (for Robustness)", 
            height=155, 
            key="dwt_extract_msg"
        )

    st.divider()
    extract_button = st.button("Extract Message", type="primary", key="dwt_extract_btn")
    
    if extract_button:
        if stego_image_file_extract is None:
            st.error("Please upload a stego-image first!")
        else:
            with st.spinner("Extracting message and calculating metrics... ⏳"):
                try:
                    file_bytes = np.asarray(bytearray(stego_image_file_extract.read()), dtype=np.uint8)
                    stego_image_rgb = cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                    
                    dwt_stego_extract = DWTSteganography(
                        wavelet=dwt_wavelet_extract,
                        level=dwt_level_extract,
                        band=dwt_band_extract,
                        delta=dwt_delta_extract,
                        embed_level=dwt_embed_level_extract,
                        robust_mode=dwt_robust_mode_extract
                    )
                    extracted_message = dwt_stego_extract.extract(stego_image_rgb, message_bit_length_extract)
                    
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
                            tester = RobustnessTester(dwt_stego_extract, original_binary_message)
                            results = tester.run_all_tests(
                                stego_image=stego_image_rgb,
                                attack_configurations=ATTACK_CONFIGURATIONS,
                                bit_lengths=message_bit_length_extract
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