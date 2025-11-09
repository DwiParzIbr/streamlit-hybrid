# ui_flows/dct_emd_ui.py

import streamlit as st
import json
import numpy as np
import pandas as pd
import cv2 
from PIL import Image
from io import BytesIO

# Impor Lokal
from constants import (
    METHOD_DCT_EMD, DEFAULT_TARGET_BIT_SIZE, 
    DCT_POSITION_PRESETS, DCT_PRESET_NAMES
)
from ui_flows.utils import generate_dummy_message_callback, reset_embed_state, make_image_grid
from methods.hybrid.dct_emd import DCTEMDHybrid, DCT_EMD_DEFAULT_PARAM
# Kelas dasar diperlukan untuk RobustnessTester
from methods.frequency.dct import DCTSteganography
from methods.spatial.emd import EMDSteganography

from metrics.impercability import SteganographyMetrics
from metrics.robustness import RobustnessTester, ATTACK_CONFIGURATIONS
from helpers.message_binary import message_to_binary

# Helper untuk menemukan nama preset default
def get_preset_name_from_list(position_list):
    position_tuples = [tuple(pos) for pos in position_list]
    for name, positions in DCT_POSITION_PRESETS.items():
        if sorted(positions) == sorted(position_tuples):
            return name
    return DCT_PRESET_NAMES[0] # Default

def draw_dct_emd_embed_tab():
    """Menampilkan UI untuk tab Embed Hybrid DCT-EMD."""
    
    default_dct_params = DCT_EMD_DEFAULT_PARAM['dct_params']
    default_emd_params = DCT_EMD_DEFAULT_PARAM['emd_params']
    default_ratio = DCT_EMD_DEFAULT_PARAM['dct_emd_ratio'][0]
    default_preset_name = get_preset_name_from_list(default_dct_params['embed_positions'])

    with st.expander("Configure Parameters"):
        hybrid_dct_ratio = st.slider(
            "DCT Ratio", min_value=0.0, max_value=1.0, 
            value=default_ratio, step=0.1, 
            key="dct_emd_embed_ratio",
            on_change=reset_embed_state, args=("dct_emd",)
        )
        st.write(f"**DCT Ratio: {hybrid_dct_ratio*100:.0f}%** | **EMD Ratio: {(1-hybrid_dct_ratio)*100:.0f}%**")
        
        with st.container(border=True):
            st.write("**DCT Parameters**")
            hybrid_quant_factor = st.number_input(
                "Quantization Factor", min_value=1, max_value=100, 
                value=default_dct_params['quant_factor'], 
                key="dct_emd_embed_quant_factor",
                on_change=reset_embed_state, args=("dct_emd",)
            )
            hybrid_preset_name = st.selectbox(
                "Embedding Position Preset", 
                options=DCT_PRESET_NAMES, 
                index=DCT_PRESET_NAMES.index(default_preset_name), 
                key="dct_emd_embed_pos_preset",
                on_change=reset_embed_state, args=("dct_emd",)
            )
        
        with st.container(border=True):
            st.write("**EMD Parameters**")
            hybrid_emd_n = st.number_input(
                "Pixel group size (n)", min_value=2, max_value=4,
                value=default_emd_params['n'], 
                key="dct_emd_embed_n",
                on_change=reset_embed_state, args=("dct_emd",)
            )

    st.subheader("Message Input")
    col1, col2 = st.columns([0.6, 0.4]) 
    with col1:
        hybrid_embed_msg = st.text_area("Your Secret Message", height=150, key="dct_emd_embed_msg", label_visibility="collapsed",
                                        on_change=reset_embed_state, args=("dct_emd",))
    with col2:
        with st.container(border=True):
            st.write("Dummy Message Helper")
            dummy_bit_len = st.number_input("Target Bit Length", min_value=8, value=DEFAULT_TARGET_BIT_SIZE, step=8, key="dct_emd_dummy_bit_len")
            st.button("Generate Dummy Message", key="dct_emd_dummy_btn", on_click=generate_dummy_message_callback, args=("dct_emd_dummy_bit_len", "dct_emd_embed_msg"))

    st.subheader("Cover Image")
    cover_image_file = st.file_uploader("Upload Cover Image", type=["png", "jpg", "bmp"], key="dct_emd_embed_img", label_visibility="collapsed",
                                        on_change=reset_embed_state, args=("dct_emd",))
        
    if cover_image_file is not None:
        st.image(cover_image_file, caption="Uploaded Cover Image", use_container_width=True)
        
    st.divider()
    embed_button = st.button("Embed Message", type="primary", key="dct_emd_embed_btn")

    if embed_button:
        if cover_image_file is None:
            st.error("Please upload a cover image first!")
            st.session_state.dct_emd_stego_image_bytes = None
            st.session_state.dct_emd_params_json = None
        else:
            with st.spinner("Embedding message, please wait... ⏳"):
                try:
                    file_bytes = np.asarray(bytearray(cover_image_file.read()), dtype=np.uint8)
                    cover_image_rgb = cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                    
                    dct_params_to_use = {
                        'quant_factor': hybrid_quant_factor,
                        'embed_positions': DCT_POSITION_PRESETS[hybrid_preset_name]
                    }
                    emd_params_to_use = {
                        'n': hybrid_emd_n
                    }
                    
                    hybrid_stego = DCTEMDHybrid(
                        dct_emd_ratio=(hybrid_dct_ratio, 1.0 - hybrid_dct_ratio),
                        dct_params=dct_params_to_use,
                        emd_params=emd_params_to_use
                    )
                    
                    stego_image_np, (dct_len, emd_len) = hybrid_stego.embed(cover_image_rgb, hybrid_embed_msg)

                    stego_image_pil = Image.fromarray(stego_image_np)
                    buffer = BytesIO()
                    stego_image_pil.save(buffer, format="PNG")
                    
                    parameters_to_save = {
                        "method": METHOD_DCT_EMD,
                        "dct_emd_ratio": (hybrid_dct_ratio, 1.0 - hybrid_dct_ratio),
                        "dct_params": dct_params_to_use,
                        "emd_params": emd_params_to_use,
                        "message_bit_lengths": (dct_len, emd_len) # Simpan tuple
                    }
                    
                    st.session_state.dct_emd_stego_image_bytes = buffer.getvalue()
                    st.session_state.dct_emd_params_json = json.dumps(parameters_to_save, indent=4, default=tuple)
                            
                except ValueError as e:
                    st.error(f"Embedding Failed: {e}")
                    st.session_state.dct_emd_stego_image_bytes = None
                    st.session_state.dct_emd_params_json = None
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    st.session_state.dct_emd_stego_image_bytes = None
                    st.session_state.dct_emd_params_json = None
    
    if st.session_state.dct_emd_stego_image_bytes is not None:
        st.subheader("Embedding Results")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.write("**Stego-Image**")
            st.image(st.session_state.dct_emd_stego_image_bytes, caption="Steganographic Image", use_container_width=True)
            st.download_button(label="Download Image", data=st.session_state.dct_emd_stego_image_bytes, file_name="dct_emd_stego_image.png", mime="image/png")
        with res_col2:
            st.write("**Parameters Used**")
            st.download_button(label="Download Parameters", data=st.session_state.dct_emd_params_json, file_name="dct_emd_parameters.json", mime="application/json")


def draw_dct_emd_extract_tab():
    """Menampilkan UI untuk tab Extract Hybrid DCT-EMD."""
    
    st.subheader("Load Parameters (Optional)")
    param_file = st.file_uploader(
        "Upload parameters.json file", 
        type=["json"], 
        key="dct_emd_extract_param_file"
    )
    
    if param_file is not None:
        try:
            param_data = json.loads(param_file.read().decode('utf-8'))
            
            if param_data.get('method') != METHOD_DCT_EMD:
                st.warning(f"File parameter ini untuk '{param_data.get('method')}', tetapi Anda memilih Hybrid.")
            
            if param_data.get('dct_emd_ratio') is not None:
                st.session_state.dct_emd_extract_dct_emd_ratio = param_data['dct_emd_ratio'][0]
            
            if param_data.get('dct_params') is not None:
                dct_p = param_data['dct_params']
                if dct_p.get('quant_factor') is not None:
                    st.session_state.dct_emd_extract_quant_factor = dct_p['quant_factor']
                if dct_p.get('embed_positions') is not None:
                    st.session_state.dct_emd_extract_embed_positions = get_preset_name_from_list(dct_p['embed_positions'])
            
            if param_data.get('emd_params') is not None:
                emd_p = param_data['emd_params']
                if emd_p.get('n') is not None:
                    st.session_state.dct_emd_extract_n = emd_p['n']
            
            if param_data.get('message_bit_lengths') is not None:
                st.session_state.dct_emd_extract_dct_bit_length = param_data['message_bit_lengths'][0]
                st.session_state.dct_emd_extract_emd_bit_length = param_data['message_bit_lengths'][1]

            st.toast("Parameters loaded successfully!")

        except Exception as e:
            st.error(f"Failed to load parameters: {e}")

    with st.expander("Configure Parameters"):
        hybrid_extract_ratio = st.slider(
            "DCT Ratio", min_value=0.0, max_value=1.0, 
            step=0.1, 
            key="dct_emd_extract_dct_emd_ratio"
        )
        st.write(f"**DCT Ratio: {hybrid_extract_ratio*100:.0f}%** | **EMD Ratio: {(1-hybrid_extract_ratio)*100:.0f}%**")
        
        with st.container(border=True):
            st.write("**DCT Parameters**")
            hybrid_extract_quant = st.number_input(
                "Quantization Factor", min_value=1, max_value=100, 
                key="dct_emd_extract_quant_factor"
            )
            hybrid_extract_preset = st.selectbox(
                "Embedding Position Preset", 
                options=DCT_PRESET_NAMES, 
                key="dct_emd_extract_embed_positions"
            )
        
        with st.container(border=True):
            st.write("**EMD Parameters**")
            hybrid_extract_n = st.number_input(
                "Pixel group size (n)", min_value=2, max_value=4,
                key="dct_emd_extract_n"
            )

    st.subheader("Upload Stego-Image")
    stego_image_file_extract = st.file_uploader(
        "Upload the image you want to extract from", 
        type=["png", "jpg", "bmp"], 
        key="dct_emd_extract_img",
        label_visibility="collapsed"
    )
    if stego_image_file_extract is not None:
        st.image(stego_image_file_extract, caption="Uploaded Stego-Image", use_container_width=True)

    st.subheader("Message Bit Lengths")
    col1, col2 = st.columns(2)
    with col1:
        dct_bit_length_extract = st.number_input(
            "DCT Bit Length", min_value=1, 
            key="dct_emd_extract_dct_bit_length"
        )
    with col2:
        emd_bit_length_extract = st.number_input(
            "EMD Bit Length", min_value=1, 
            key="dct_emd_extract_emd_bit_length"
        )
    
    st.subheader("Optional Inputs for Metrics")
    col1_opt, col2_opt = st.columns(2)
    with col1_opt:
        optional_cover_image = st.file_uploader(
            "Original Cover Image (for Imperceptibility)", 
            type=["png", "jpg", "bmp"], 
            key="dct_emd_extract_cover"
        )
    with col2_opt:
        optional_original_message = st.text_area(
            "Original Message (for Robustness)", 
            height=155, 
            key="dct_emd_extract_msg"
        )

    st.divider()
    extract_button = st.button("Extract Message", type="primary", key="dct_emd_extract_btn")
    
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
                    emd_params_ext = {
                        'n': hybrid_extract_n
                    }
                    
                    hybrid_stego_extract = DCTEMDHybrid(
                        dct_emd_ratio=(hybrid_extract_ratio, 1.0 - hybrid_extract_ratio),
                        dct_params=dct_params_ext,
                        emd_params=emd_params_ext
                    )
                    
                    bit_lengths_tuple = (dct_bit_length_extract, emd_bit_length_extract)
                    
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