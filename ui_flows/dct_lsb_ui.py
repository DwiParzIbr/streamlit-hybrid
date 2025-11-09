# ui_flows/dct_lsb_ui.py

import streamlit as st
import json
import numpy as np
import pandas as pd
import cv2 
from PIL import Image
from io import BytesIO

# Impor Lokal
from constants import (
    METHOD_DCT_LSB, DEFAULT_TARGET_BIT_SIZE, # <--- DIPERBARUI
    DCT_POSITION_PRESETS, DCT_PRESET_NAMES
)
from ui_flows.utils import generate_dummy_message_callback, reset_embed_state, make_image_grid
from methods.hybrid.dct_lsb import DCTLSBHybrid, DCT_LSB_DEFAULT_PARAM
from methods.frequency.dct import DCTSteganography
from methods.spatial.lsb import LSBSteganography

from metrics.impercability import SteganographyMetrics
from metrics.robustness import RobustnessTester, ATTACK_CONFIGURATIONS
from helpers.message_binary import message_to_binary

# Helper untuk menemukan nama preset default
def get_preset_name_from_list(position_list):
    # Konversi list yang dimuat (list of list) ke list of tuple untuk perbandingan
    position_tuples = [tuple(pos) for pos in position_list]
    for name, positions in DCT_POSITION_PRESETS.items():
        if sorted(positions) == sorted(position_tuples):
            return name
    return DCT_PRESET_NAMES[0] # Default

# --- PERUBAHAN NAMA FUNGSI ---
def draw_dct_lsb_embed_tab():
    """Menampilkan UI untuk tab Embed Hybrid."""
    
    default_dct_params = DCT_LSB_DEFAULT_PARAM['dct_params']
    default_lsb_params = DCT_LSB_DEFAULT_PARAM['lsb_params']
    default_ratio = DCT_LSB_DEFAULT_PARAM['dct_lsb_ratio'][0]
    default_preset_name = get_preset_name_from_list(default_dct_params['embed_positions'])

    with st.expander("Configure Parameters"):
        hybrid_dct_ratio = st.slider(
            "DCT Ratio", min_value=0.0, max_value=1.0, 
            value=default_ratio, step=0.1, 
            key="dct_lsb_embed_ratio", # <--- PERUBAHAN KEY
            on_change=reset_embed_state, args=("dct_lsb",) # <--- PERUBAHAN ARGS
        )
        st.write(f"**DCT Ratio: {hybrid_dct_ratio*100:.0f}%** | **LSB Ratio: {(1-hybrid_dct_ratio)*100:.0f}%**")
        
        with st.container(border=True):
            st.write("**DCT Parameters**")
            hybrid_quant_factor = st.number_input(
                "Quantization Factor", min_value=1, max_value=100, 
                value=default_dct_params['quant_factor'], 
                key="dct_lsb_embed_quant_factor", # <--- PERUBAHAN KEY
                on_change=reset_embed_state, args=("dct_lsb",) # <--- PERUBAHAN ARGS
            )
            hybrid_preset_name = st.selectbox(
                "Embedding Position Preset", 
                options=DCT_PRESET_NAMES, 
                index=DCT_PRESET_NAMES.index(default_preset_name), 
                key="dct_lsb_embed_pos_preset", # <--- PERUBAHAN KEY
                on_change=reset_embed_state, args=("dct_lsb",) # <--- PERUBAHAN ARGS
            )
        
        with st.container(border=True):
            st.write("**LSB Parameters**")
            hybrid_lsb_bits = st.number_input(
                "Bits Per Channel (e.g., 1-8)", min_value=1, max_value=8, 
                value=default_lsb_params['bits_per_channel'], 
                key="dct_lsb_embed_bits_per_channel", # <--- PERUBAHAN KEY
                on_change=reset_embed_state, args=("dct_lsb",) # <--- PERUBAHAN ARGS
            )

    st.subheader("Message Input")
    col1, col2 = st.columns([0.6, 0.4]) 
    with col1:
        hybrid_embed_msg = st.text_area("Your Secret Message", height=150, key="dct_lsb_embed_msg", label_visibility="collapsed", # <--- PERUBAHAN KEY
                                        on_change=reset_embed_state, args=("dct_lsb",)) # <--- PERUBAHAN ARGS
    with col2:
        with st.container(border=True):
            st.write("Dummy Message Helper")
            dummy_bit_len = st.number_input("Target Bit Length", min_value=8, value=DEFAULT_TARGET_BIT_SIZE, step=8, key="dct_lsb_dummy_bit_len") # <--- PERUBAHAN KEY
            st.button("Generate Dummy Message", key="dct_lsb_dummy_btn", on_click=generate_dummy_message_callback, args=("dct_lsb_dummy_bit_len", "dct_lsb_embed_msg")) # <--- PERUBAHAN KEY

    st.subheader("Cover Image")
    cover_image_file = st.file_uploader("Upload Cover Image", type=["png", "jpg", "bmp"], key="dct_lsb_embed_img", label_visibility="collapsed", # <--- PERUBAHAN KEY
                                        on_change=reset_embed_state, args=("dct_lsb",)) # <--- PERUBAHAN ARGS
        
    if cover_image_file is not None:
        st.image(cover_image_file, caption="Uploaded Cover Image", use_container_width=True)
        
    st.divider()
    embed_button = st.button("Embed Message", type="primary", key="dct_lsb_embed_btn") # <--- PERUBAHAN KEY

    if embed_button:
        if cover_image_file is None:
            st.error("Please upload a cover image first!")
            st.session_state.dct_lsb_stego_image_bytes = None # <--- PERUBAHAN KEY
            st.session_state.dct_lsb_params_json = None # <--- PERUBAHAN KEY
        else:
            with st.spinner("Embedding message, please wait... ⏳"):
                try:
                    file_bytes = np.asarray(bytearray(cover_image_file.read()), dtype=np.uint8)
                    cover_image_rgb = cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                    
                    dct_params_to_use = {
                        'quant_factor': hybrid_quant_factor,
                        'embed_positions': DCT_POSITION_PRESETS[hybrid_preset_name]
                    }
                    lsb_params_to_use = {
                        'bits_per_channel': hybrid_lsb_bits
                    }
                    
                    hybrid_stego = DCTLSBHybrid(
                        dct_lsb_ratio=(hybrid_dct_ratio, 1.0 - hybrid_dct_ratio),
                        dct_params=dct_params_to_use,
                        lsb_params=lsb_params_to_use
                    )
                    
                    stego_image_np, (dct_len, lsb_len) = hybrid_stego.embed(cover_image_rgb, hybrid_embed_msg)

                    stego_image_pil = Image.fromarray(stego_image_np)
                    buffer = BytesIO()
                    stego_image_pil.save(buffer, format="PNG")
                    
                    parameters_to_save = {
                        "method": METHOD_DCT_LSB, # <--- DIPERBARUI
                        "dct_lsb_ratio": (hybrid_dct_ratio, 1.0 - hybrid_dct_ratio),
                        "dct_params": dct_params_to_use,
                        "lsb_params": lsb_params_to_use,
                        "message_bit_lengths": (dct_len, lsb_len)
                    }
                    
                    st.session_state.dct_lsb_stego_image_bytes = buffer.getvalue() # <--- PERUBAHAN KEY
                    st.session_state.dct_lsb_params_json = json.dumps(parameters_to_save, indent=4, default=tuple) # <--- PERUBAHAN KEY
                            
                except ValueError as e:
                    st.error(f"Embedding Failed: {e}")
                    st.session_state.dct_lsb_stego_image_bytes = None # <--- PERUBAHAN KEY
                    st.session_state.dct_lsb_params_json = None # <--- PERUBAHAN KEY
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    st.session_state.dct_lsb_stego_image_bytes = None # <--- PERUBAHAN KEY
                    st.session_state.dct_lsb_params_json = None # <--- PERUBAHAN KEY
    
    if st.session_state.dct_lsb_stego_image_bytes is not None: # <--- PERUBAHAN KEY
        st.subheader("Embedding Results")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.write("**Stego-Image**")
            st.image(st.session_state.dct_lsb_stego_image_bytes, caption="Steganographic Image", use_container_width=True) # <--- PERUBAHAN KEY
            st.download_button(label="Download Image", data=st.session_state.dct_lsb_stego_image_bytes, file_name="dct_lsb_stego_image.png", mime="image/png") # <--- DIPERBARUI
        with res_col2:
            st.write("**Parameters Used**")
            st.download_button(label="Download Parameters", data=st.session_state.dct_lsb_params_json, file_name="dct_lsb_parameters.json", mime="application/json") # <--- DIPERBARUI


# --- PERUBAHAN NAMA FUNGSI ---
def draw_dct_lsb_extract_tab():
    """Menampilkan UI untuk tab Extract Hybrid."""
    
    st.subheader("Load Parameters (Optional)")
    param_file = st.file_uploader(
        "Upload parameters.json file", 
        type=["json"], 
        key="dct_lsb_extract_param_file" # <--- PERUBAHAN KEY
    )
    
    if param_file is not None:
        try:
            param_data = json.loads(param_file.read().decode('utf-8'))
            
            if param_data.get('method') != METHOD_DCT_LSB: # <--- DIPERBARUI
                st.warning(f"File parameter ini untuk '{param_data.get('method')}', tetapi Anda memilih Hybrid.")
            
            if param_data.get('dct_lsb_ratio') is not None:
                st.session_state.dct_lsb_extract_dct_lsb_ratio = param_data['dct_lsb_ratio'][0] # <--- PERUBAHAN KEY
            
            if param_data.get('dct_params') is not None:
                dct_p = param_data['dct_params']
                if dct_p.get('quant_factor') is not None:
                    st.session_state.dct_lsb_extract_quant_factor = dct_p['quant_factor'] # <--- PERUBAHAN KEY
                if dct_p.get('embed_positions') is not None:
                    st.session_state.dct_lsb_extract_embed_positions = get_preset_name_from_list(dct_p['embed_positions']) # <--- PERUBAHAN KEY
            
            if param_data.get('lsb_params') is not None:
                lsb_p = param_data['lsb_params']
                if lsb_p.get('bits_per_channel') is not None:
                    st.session_state.dct_lsb_extract_bits_per_channel = lsb_p['bits_per_channel'] # <--- PERUBAHAN KEY
            
            if param_data.get('message_bit_lengths') is not None:
                st.session_state.dct_lsb_extract_dct_bit_length = param_data['message_bit_lengths'][0] # <--- PERUBAHAN KEY
                st.session_state.dct_lsb_extract_lsb_bit_length = param_data['message_bit_lengths'][1] # <--- PERUBAHAN KEY

            st.toast("Parameters loaded successfully!")

        except Exception as e:
            st.error(f"Failed to load parameters: {e}")

    with st.expander("Configure Parameters"):
        hybrid_extract_ratio = st.slider(
            "DCT Ratio", min_value=0.0, max_value=1.0, 
            step=0.1, 
            key="dct_lsb_extract_dct_lsb_ratio" # <--- PERUBAHAN KEY
        )
        st.write(f"**DCT Ratio: {hybrid_extract_ratio*100:.0f}%** | **LSB Ratio: {(1-hybrid_extract_ratio)*100:.0f}%**")
        
        with st.container(border=True):
            st.write("**DCT Parameters**")
            hybrid_extract_quant = st.number_input(
                "Quantization Factor", min_value=1, max_value=100, 
                key="dct_lsb_extract_quant_factor" # <--- PERUBAHAN KEY
            )
            hybrid_extract_preset = st.selectbox(
                "Embedding Position Preset", 
                options=DCT_PRESET_NAMES, 
                key="dct_lsb_extract_embed_positions" # <--- PERUBAHAN KEY
            )
        
        with st.container(border=True):
            st.write("**LSB Parameters**")
            hybrid_extract_bits = st.number_input(
                "Bits Per Channel (e.g., 1-8)", min_value=1, max_value=8, 
                key="dct_lsb_extract_bits_per_channel" # <--- PERUBAHAN KEY
            )

    st.subheader("Upload Stego-Image")
    stego_image_file_extract = st.file_uploader(
        "Upload the image you want to extract from", 
        type=["png", "jpg", "bmp"], 
        key="dct_lsb_extract_img", # <--- PERUBAHAN KEY
        label_visibility="collapsed"
    )
    if stego_image_file_extract is not None:
        st.image(stego_image_file_extract, caption="Uploaded Stego-Image", use_container_width=True)

    st.subheader("Message Bit Lengths")
    col1, col2 = st.columns(2)
    with col1:
        dct_bit_length_extract = st.number_input(
            "DCT Bit Length", min_value=1, 
            key="dct_lsb_extract_dct_bit_length" # <--- PERUBAHAN KEY
        )
    with col2:
        lsb_bit_length_extract = st.number_input(
            "LSB Bit Length", min_value=1, 
            key="dct_lsb_extract_lsb_bit_length" # <--- PERUBAHAN KEY
        )
    
    st.subheader("Optional Inputs for Metrics")
    col1_opt, col2_opt = st.columns(2)
    with col1_opt:
        optional_cover_image = st.file_uploader(
            "Original Cover Image (for Imperceptibility)", 
            type=["png", "jpg", "bmp"], 
            key="dct_lsb_extract_cover" # <--- PERUBAHAN KEY
        )
    with col2_opt:
        optional_original_message = st.text_area(
            "Original Message (for Robustness)", 
            height=155, 
            key="dct_lsb_extract_msg" # <--- PERUBAHAN KEY
        )

    st.divider()
    extract_button = st.button("Extract Message", type="primary", key="dct_lsb_extract_btn") # <--- PERUBAHAN KEY
    
    if extract_button:
        if stego_image_file_extract is None:
            st.error("Please upload a stego-image first!")
        else:
            with st.spinner("Extracting message and calculating metrics... ⏳"):
                try:
                    file_bytes = np.asarray(bytearray(stego_image_file_extract.read()), dtype=np.uint8)
                    stego_image_rgb = cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                    
                    dct_params_ext = {
                        'quant_factor': hybrid_extract_quant,
                        'embed_positions': DCT_POSITION_PRESETS[hybrid_extract_preset]
                    }
                    lsb_params_ext = {
                        'bits_per_channel': hybrid_extract_bits
                    }
                    hybrid_stego_extract = DCTLSBHybrid(
                        dct_lsb_ratio=(hybrid_extract_ratio, 1.0 - hybrid_extract_ratio),
                        dct_params=dct_params_ext,
                        lsb_params=lsb_params_ext
                    )
                    
                    bit_lengths_tuple = (dct_bit_length_extract, lsb_bit_length_extract)
                    
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