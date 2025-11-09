# ui_flows/fft_lsb_ui.py

import streamlit as st
import json
import numpy as np
import pandas as pd
import cv2 
from PIL import Image
from io import BytesIO

# Impor Lokal
from constants import (
    METHOD_FFT_LSB, DEFAULT_TARGET_BIT_SIZE
)
from ui_flows.utils import generate_dummy_message_callback, reset_embed_state, make_image_grid
from methods.hybrid.fft_lsb import FFTLSBHybrid, FFT_LSB_DEFAULT_PARAM
# Kelas dasar diperlukan untuk RobustnessTester
from methods.frequency.fft import FFTSteganography
from methods.spatial.lsb import LSBSteganography

from metrics.impercability import SteganographyMetrics
from metrics.robustness import RobustnessTester, ATTACK_CONFIGURATIONS
from helpers.message_binary import message_to_binary

# Daftar channel YCrCb
CHANNEL_LIST = ["Y", "Cr", "Cb"]

def draw_fft_lsb_embed_tab():
    """Menampilkan UI untuk tab Embed Hybrid FFT-LSB."""
    
    default_fft_params = FFT_LSB_DEFAULT_PARAM['fft_params']
    default_lsb_params = FFT_LSB_DEFAULT_PARAM['lsb_params']
    default_ratio = FFT_LSB_DEFAULT_PARAM['fft_lsb_ratio'][0]

    with st.expander("Configure Parameters"):
        hybrid_fft_ratio = st.slider(
            "FFT Ratio", min_value=0.0, max_value=1.0, 
            value=default_ratio, step=0.1, 
            key="fft_lsb_embed_ratio",
            on_change=reset_embed_state, args=("fft_lsb",)
        )
        st.write(f"**FFT Ratio: {hybrid_fft_ratio*100:.0f}%** | **LSB Ratio: {(1-hybrid_fft_ratio)*100:.0f}%**")
        
        with st.container(border=True):
            st.write("**FFT Parameters**")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                r_in = st.slider(
                    "Inner Radius (r_in)", 0.0, 1.0, 
                    value=default_fft_params['r_in'], 
                    key="fft_lsb_embed_r_in",
                    on_change=reset_embed_state, args=("fft_lsb",)
                )
                r_out = st.slider(
                    "Outer Radius (r_out)", 0.0, 1.0, 
                    value=default_fft_params['r_out'], 
                    key="fft_lsb_embed_r_out",
                    on_change=reset_embed_state, args=("fft_lsb",)
                )
                header_repeat = st.number_input(
                    "Header Repetitions", min_value=1, step=2, 
                    value=default_fft_params['header_repeat'], 
                    key="fft_lsb_embed_header_repeat",
                    on_change=reset_embed_state, args=("fft_lsb",)
                )
                payload_repeat = st.number_input(
                    "Payload Repetitions", min_value=1, step=2, 
                    value=default_fft_params['payload_repeat'], 
                    key="fft_lsb_embed_payload_repeat",
                    on_change=reset_embed_state, args=("fft_lsb",)
                )
            with p_col2:
                header_channel = st.selectbox(
                    "Header Channel", CHANNEL_LIST, 
                    index=CHANNEL_LIST.index(default_fft_params['header_channel']),
                    key="fft_lsb_embed_header_channel",
                    on_change=reset_embed_state, args=("fft_lsb",)
                )
                payload_channel = st.selectbox(
                    "Payload Channel", CHANNEL_LIST, 
                    index=CHANNEL_LIST.index(default_fft_params['payload_channel']),
                    key="fft_lsb_embed_payload_channel",
                    on_change=reset_embed_state, args=("fft_lsb",)
                )
                mag_min_boost = st.number_input(
                    "Magnitude Boost", min_value=0.0, 
                    value=default_fft_params['mag_min_boost'], 
                    key="fft_lsb_embed_mag_min_boost",
                    on_change=reset_embed_state, args=("fft_lsb",)
                )
        
        with st.container(border=True):
            st.write("**LSB Parameters**")
            hybrid_lsb_bits = st.number_input(
                "Bits Per Channel (e.g., 1-8)", min_value=1, max_value=8, 
                value=default_lsb_params['bits_per_channel'], 
                key="fft_lsb_embed_bits_per_channel",
                on_change=reset_embed_state, args=("fft_lsb",)
            )

    st.subheader("Message Input")
    col1, col2 = st.columns([0.6, 0.4]) 
    with col1:
        hybrid_embed_msg = st.text_area("Your Secret Message", height=150, key="fft_lsb_embed_msg", label_visibility="collapsed",
                                        on_change=reset_embed_state, args=("fft_lsb",))
    with col2:
        with st.container(border=True):
            st.write("Dummy Message Helper")
            dummy_bit_len = st.number_input("Target Bit Length", min_value=8, value=DEFAULT_TARGET_BIT_SIZE, step=8, key="fft_lsb_dummy_bit_len")
            st.button("Generate Dummy Message", key="fft_lsb_dummy_btn", on_click=generate_dummy_message_callback, args=("fft_lsb_dummy_bit_len", "fft_lsb_embed_msg"))

    st.subheader("Cover Image")
    cover_image_file = st.file_uploader("Upload Cover Image", type=["png", "jpg", "bmp"], key="fft_lsb_embed_img", label_visibility="collapsed",
                                        on_change=reset_embed_state, args=("fft_lsb",))
        
    if cover_image_file is not None:
        st.image(cover_image_file, caption="Uploaded Cover Image", use_container_width=True)
        
    st.divider()
    embed_button = st.button("Embed Message", type="primary", key="fft_lsb_embed_btn")

    if embed_button:
        if cover_image_file is None:
            st.error("Please upload a cover image first!")
            st.session_state.fft_lsb_stego_image_bytes = None
            st.session_state.fft_lsb_params_json = None
        else:
            with st.spinner("Embedding message, please wait... ⏳"):
                try:
                    file_bytes = np.asarray(bytearray(cover_image_file.read()), dtype=np.uint8)
                    cover_image_rgb = cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                    
                    fft_params_to_use = {
                        'r_in': r_in,
                        'r_out': r_out,
                        'header_repeat': header_repeat,
                        'payload_repeat': payload_repeat,
                        'header_channel': header_channel,
                        'payload_channel': payload_channel,
                        'mag_min_boost': mag_min_boost,
                        'color_order': 'RGB'
                    }
                    lsb_params_to_use = {
                        'bits_per_channel': hybrid_lsb_bits
                    }
                    
                    hybrid_stego = FFTLSBHybrid(
                        fft_lsb_ratio=(hybrid_fft_ratio, 1.0 - hybrid_fft_ratio),
                        fft_params=fft_params_to_use,
                        lsb_params=lsb_params_to_use
                    )
                    
                    stego_image_np, (fft_len, lsb_len) = hybrid_stego.embed(cover_image_rgb, hybrid_embed_msg)

                    stego_image_pil = Image.fromarray(stego_image_np)
                    buffer = BytesIO()
                    stego_image_pil.save(buffer, format="PNG")
                    
                    parameters_to_save = {
                        "method": METHOD_FFT_LSB,
                        "fft_lsb_ratio": (hybrid_fft_ratio, 1.0 - hybrid_fft_ratio),
                        "fft_params": fft_params_to_use,
                        "lsb_params": lsb_params_to_use,
                        "message_bit_lengths": (fft_len, lsb_len)
                    }
                    
                    st.session_state.fft_lsb_stego_image_bytes = buffer.getvalue()
                    st.session_state.fft_lsb_params_json = json.dumps(parameters_to_save, indent=4, default=tuple)
                            
                except ValueError as e:
                    st.error(f"Embedding Failed: {e}")
                    st.session_state.fft_lsb_stego_image_bytes = None
                    st.session_state.fft_lsb_params_json = None
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    st.session_state.fft_lsb_stego_image_bytes = None
                    st.session_state.fft_lsb_params_json = None
    
    if st.session_state.fft_lsb_stego_image_bytes is not None:
        st.subheader("Embedding Results")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.write("**Stego-Image**")
            st.image(st.session_state.fft_lsb_stego_image_bytes, caption="Steganographic Image", use_container_width=True)
            st.download_button(label="Download Image", data=st.session_state.fft_lsb_stego_image_bytes, file_name="fft_lsb_stego_image.png", mime="image/png")
        with res_col2:
            st.write("**Parameters Used**")
            st.download_button(label="Download Parameters", data=st.session_state.fft_lsb_params_json, file_name="fft_lsb_parameters.json", mime="application/json")


def draw_fft_lsb_extract_tab():
    """Menampilkan UI untuk tab Extract Hybrid FFT-LSB."""
    
    st.subheader("Load Parameters (Optional)")
    param_file = st.file_uploader(
        "Upload parameters.json file", 
        type=["json"], 
        key="fft_lsb_extract_param_file"
    )
    
    if param_file is not None:
        try:
            param_data = json.loads(param_file.read().decode('utf-8'))
            
            if param_data.get('method') != METHOD_FFT_LSB:
                st.warning(f"File parameter ini untuk '{param_data.get('method')}', tetapi Anda memilih Hybrid.")
            
            if param_data.get('fft_lsb_ratio') is not None:
                st.session_state.fft_lsb_extract_fft_lsb_ratio = param_data['fft_lsb_ratio'][0]
            
            if param_data.get('fft_params') is not None:
                fft_p = param_data['fft_params']
                if fft_p.get('r_in') is not None: st.session_state.fft_lsb_extract_r_in = fft_p['r_in']
                if fft_p.get('r_out') is not None: st.session_state.fft_lsb_extract_r_out = fft_p['r_out']
                if fft_p.get('header_repeat') is not None: st.session_state.fft_lsb_extract_header_repeat = fft_p['header_repeat']
                if fft_p.get('payload_repeat') is not None: st.session_state.fft_lsb_extract_payload_repeat = fft_p['payload_repeat']
                if fft_p.get('header_channel') is not None: st.session_state.fft_lsb_extract_header_channel = fft_p['header_channel']
                if fft_p.get('payload_channel') is not None: st.session_state.fft_lsb_extract_payload_channel = fft_p['payload_channel']
                if fft_p.get('mag_min_boost') is not None: st.session_state.fft_lsb_extract_mag_min_boost = fft_p['mag_min_boost']

            if param_data.get('lsb_params') is not None:
                lsb_p = param_data['lsb_params']
                if lsb_p.get('bits_per_channel') is not None:
                    st.session_state.fft_lsb_extract_bits_per_channel = lsb_p['bits_per_channel']
            
            if param_data.get('message_bit_lengths') is not None:
                st.session_state.fft_lsb_extract_fft_bit_length = param_data['message_bit_lengths'][0]
                st.session_state.fft_lsb_extract_lsb_bit_length = param_data['message_bit_lengths'][1]

            st.toast("Parameters loaded successfully!")

        except Exception as e:
            st.error(f"Failed to load parameters: {e}")

    with st.expander("Configure Parameters"):
        hybrid_extract_ratio = st.slider(
            "FFT Ratio", min_value=0.0, max_value=1.0, 
            step=0.1, 
            key="fft_lsb_extract_fft_lsb_ratio"
        )
        st.write(f"**FFT Ratio: {hybrid_extract_ratio*100:.0f}%** | **LSB Ratio: {(1-hybrid_extract_ratio)*100:.0f}%**")
        
        with st.container(border=True):
            st.write("**FFT Parameters**")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                r_in_extract = st.slider("Inner Radius (r_in)", 0.0, 1.0, key="fft_lsb_extract_r_in")
                r_out_extract = st.slider("Outer Radius (r_out)", 0.0, 1.0, key="fft_lsb_extract_r_out")
                header_repeat_extract = st.number_input("Header Repetitions", min_value=1, step=2, key="fft_lsb_extract_header_repeat")
                payload_repeat_extract = st.number_input("Payload Repetitions", min_value=1, step=2, key="fft_lsb_extract_payload_repeat")
            with p_col2:
                header_channel_extract = st.selectbox("Header Channel", CHANNEL_LIST, key="fft_lsb_extract_header_channel")
                payload_channel_extract = st.selectbox("Payload Channel", CHANNEL_LIST, key="fft_lsb_extract_payload_channel")
                mag_min_boost_extract = st.number_input("Magnitude Boost", min_value=0.0, key="fft_lsb_extract_mag_min_boost")

        with st.container(border=True):
            st.write("**LSB Parameters**")
            hybrid_extract_bits = st.number_input(
                "Bits Per Channel (e.g., 1-8)", min_value=1, max_value=8, 
                key="fft_lsb_extract_bits_per_channel"
            )

    st.subheader("Upload Stego-Image")
    stego_image_file_extract = st.file_uploader(
        "Upload the image you want to extract from", 
        type=["png", "jpg", "bmp"], 
        key="fft_lsb_extract_img",
        label_visibility="collapsed"
    )
    if stego_image_file_extract is not None:
        st.image(stego_image_file_extract, caption="Uploaded Stego-Image", use_container_width=True)

    st.subheader("Message Bit Lengths")
    col1, col2 = st.columns(2)
    with col1:
        fft_bit_length_extract = st.number_input(
            "FFT Bit Length", min_value=1, 
            key="fft_lsb_extract_fft_bit_length"
        )
    with col2:
        lsb_bit_length_extract = st.number_input(
            "LSB Bit Length", min_value=1, 
            key="fft_lsb_extract_lsb_bit_length"
        )
    
    st.subheader("Optional Inputs for Metrics")
    col1_opt, col2_opt = st.columns(2)
    with col1_opt:
        optional_cover_image = st.file_uploader(
            "Original Cover Image (for Imperceptibility)", 
            type=["png", "jpg", "bmp"], 
            key="fft_lsb_extract_cover"
        )
    with col2_opt:
        optional_original_message = st.text_area(
            "Original Message (for Robustness)", 
            height=155, 
            key="fft_lsb_extract_msg"
        )

    st.divider()
    extract_button = st.button("Extract Message", type="primary", key="fft_lsb_extract_btn")
    
    if extract_button:
        if stego_image_file_extract is None:
            st.error("Please upload a stego-image first!")
        else:
            with st.spinner("Extracting message and calculating metrics... ⏳"):
                try:
                    file_bytes = np.asarray(bytearray(stego_image_file_extract.read()), dtype=np.uint8)
                    stego_image_rgb = cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                    
                    fft_params_ext = {
                        'r_in': r_in_extract,
                        'r_out': r_out_extract,
                        'header_repeat': header_repeat_extract,
                        'payload_repeat': payload_repeat_extract,
                        'header_channel': header_channel_extract,
                        'payload_channel': payload_channel_extract,
                        'mag_min_boost': mag_min_boost_extract,
                        'color_order': 'RGB'
                    }
                    lsb_params_ext = {
                        'bits_per_channel': hybrid_extract_bits
                    }
                    
                    hybrid_stego_extract = FFTLSBHybrid(
                        fft_lsb_ratio=(hybrid_extract_ratio, 1.0 - hybrid_extract_ratio),
                        fft_params=fft_params_ext,
                        lsb_params=lsb_params_ext
                    )
                    
                    bit_lengths_tuple = (fft_bit_length_extract, lsb_bit_length_extract)
                    
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