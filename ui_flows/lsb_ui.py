# ui_flows/lsb_ui.py

import streamlit as st
import json
import numpy as np
import pandas as pd
import cv2 
from PIL import Image
from io import BytesIO

# Impor Lokal
from constants import METHOD_LSB, DEFAULT_TARGET_BIT_SIZE
from ui_flows.utils import generate_dummy_message_callback, reset_embed_state
from methods.spatial.lsb import LSBSteganography, LSB_DEFAULT_PARAM
from metrics.impercability import SteganographyMetrics
from metrics.robustness import RobustnessTester, ATTACK_CONFIGURATIONS
from helpers.message_binary import message_to_binary
from ui_flows.utils import make_image_grid

def draw_lsb_embed_tab():
    """Menampilkan UI untuk tab Embed LSB."""
    
    with st.expander("Configure Parameters"):
        bits_per_channel_embed = st.number_input(
            "Bits Per Channel (e.g., 1-8)", min_value=1, max_value=8, 
            value=LSB_DEFAULT_PARAM['bits_per_channel'], 
            key="lsb_embed_bits_per_channel",
            on_change=reset_embed_state, args=("lsb",)
        )

    st.subheader("Message Input")
    col1, col2 = st.columns([0.6, 0.4]) 
    with col1:
        lsb_embed_msg = st.text_area("Your Secret Message", height=150, key="lsb_embed_msg", label_visibility="collapsed",
                                     on_change=reset_embed_state, args=("lsb",))
    with col2:
        with st.container(border=True):
            st.write("Dummy Message Helper")
            dummy_bit_len = st.number_input("Target Bit Length", min_value=8, value=DEFAULT_TARGET_BIT_SIZE, step=8, key="lsb_dummy_bit_len")
            st.button("Generate Dummy Message", key="lsb_dummy_btn", on_click=generate_dummy_message_callback, args=("lsb_dummy_bit_len", "lsb_embed_msg"))
    
    st.subheader("Cover Image")
    cover_image_file = st.file_uploader("Upload Cover Image", type=["png", "jpg", "bmp"], key="lsb_embed_img", label_visibility="collapsed",
                                        on_change=reset_embed_state, args=("lsb",))
        
    if cover_image_file is not None:
        st.image(cover_image_file, caption="Uploaded Cover Image", use_container_width=True)
        
    st.divider()
    embed_button = st.button("Embed Message", type="primary", key="lsb_embed_btn")
    
    if embed_button:
        if cover_image_file is None:
            st.error("Please upload a cover image first!")
            st.session_state.lsb_stego_image_bytes = None 
            st.session_state.lsb_params_json = None
        else:
            # --- PERUBAHAN DI SINI: Menambahkan st.spinner ---
            with st.spinner("Embedding message, please wait... ⏳"):
                try:
                    file_bytes = np.asarray(bytearray(cover_image_file.read()), dtype=np.uint8)
                    cover_image_rgb = cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                    
                    lsb_stego = LSBSteganography(bits_per_channel=bits_per_channel_embed)
                    stego_image_np, final_bit_length = lsb_stego.embed(cover_image_rgb, lsb_embed_msg)

                    stego_image_pil = Image.fromarray(stego_image_np)
                    buffer = BytesIO()
                    stego_image_pil.save(buffer, format="PNG")
                    
                    parameters_to_save = {
                        "method": METHOD_LSB, 
                        "bits_per_channel": bits_per_channel_embed,
                        "message_bit_length": final_bit_length
                    }
                    
                    st.session_state.lsb_stego_image_bytes = buffer.getvalue()
                    st.session_state.lsb_params_json = json.dumps(parameters_to_save, indent=4)
                            
                except ValueError as e:
                    st.error(f"Embedding Failed: {e}")
                    st.session_state.lsb_stego_image_bytes = None
                    st.session_state.lsb_params_json = None
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    st.session_state.lsb_stego_image_bytes = None
                    st.session_state.lsb_params_json = None
    
    if st.session_state.lsb_stego_image_bytes is not None:
        st.subheader("Embedding Results")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.write("**Stego-Image**")
            st.image(st.session_state.lsb_stego_image_bytes, caption="Steganographic Image", use_container_width=True)
            st.download_button(
                label="Download Image",
                data=st.session_state.lsb_stego_image_bytes, 
                file_name="lsb_stego_image.png",
                mime="image/png"
            )
        with res_col2:
            st.write("**Parameters Used**")
            st.download_button(
                label="Download Parameters",
                data=st.session_state.lsb_params_json,
                file_name="lsb_parameters.json",
                mime="application/json"
            )

def draw_lsb_extract_tab():
    """Menampilkan UI untuk tab Extract LSB."""
    
    st.subheader("Load Parameters (Optional)")
    param_file = st.file_uploader(
        "Upload parameters.json file", 
        type=["json"], 
        key="lsb_extract_param_file"
    )
    
    if param_file is not None:
        try:
            param_data = json.loads(param_file.read().decode('utf-8'))
            
            if param_data.get('method') != METHOD_LSB:
                st.warning(f"File parameter ini untuk '{param_data.get('method')}', tetapi Anda memilih LSB.")
            
            loaded_bits = param_data.get('bits_per_channel')
            loaded_bit_length = param_data.get('message_bit_length')

            if loaded_bits is not None:
                st.session_state.lsb_extract_bits_per_channel = loaded_bits
            
            if loaded_bit_length is not None:
                st.session_state.lsb_extract_bit_length = loaded_bit_length

            st.toast("Parameters loaded successfully!")

        except Exception as e:
            st.error(f"Failed to load parameters: {e}")

    with st.expander("Configure Parameters"):
        bits_per_channel_extract = st.number_input(
            "Bits Per Channel (e.g., 1-8)", 
            min_value=1, max_value=8, 
            key="lsb_extract_bits_per_channel"
        )

    st.subheader("Upload Stego-Image")
    stego_image_file_extract = st.file_uploader(
        "Upload the image you want to extract from", 
        type=["png", "jpg", "bmp"], 
        key="lsb_extract_img", 
        label_visibility="collapsed"
    )
    if stego_image_file_extract is not None:
        st.image(stego_image_file_extract, caption="Uploaded Stego-Image", use_container_width=True)

    st.subheader("Message Bit Length")
    message_bit_length_extract = st.number_input(
        "Enter the bit length of the secret message", 
        min_value=1, 
        key="lsb_extract_bit_length"
    )
    
    st.subheader("Optional Inputs for Metrics")
    col1, col2 = st.columns(2)
    with col1:
        optional_cover_image = st.file_uploader(
            "Original Cover Image (for Imperceptibility)", 
            type=["png", "jpg", "bmp"], 
            key="lsb_extract_cover"
        )
    with col2:
        optional_original_message = st.text_area(
            "Original Message (for Robustness)", 
            height=155, 
            key="lsb_extract_msg"
        )

    st.divider()
    extract_button = st.button("Extract Message", type="primary", key="lsb_extract_btn")

    if extract_button:
        if stego_image_file_extract is None:
            st.error("Please upload a stego-image first!")
        else:
            # --- PERUBAHAN DI SINI: Menambahkan st.spinner ---
            with st.spinner("Extracting message and calculating metrics... ⏳"):
                try:
                    file_bytes = np.asarray(bytearray(stego_image_file_extract.read()), dtype=np.uint8)
                    stego_image_rgb = cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                    
                    lsb_stego_extract = LSBSteganography(bits_per_channel=bits_per_channel_extract)
                    extracted_message = lsb_stego_extract.extract(stego_image_rgb, message_bit_length_extract)
                    
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
                            tester = RobustnessTester(lsb_stego_extract, original_binary_message)
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
            # --- AKHIR BLOK SPINNER ---