# ui_flows/dct_ui.py

import streamlit as st
import json
import numpy as np
import pandas as pd
import cv2 
from PIL import Image
from io import BytesIO

# Impor Lokal
from constants import METHOD_DCT, DEFAULT_TARGET_BIT_SIZE, DCT_PRESET_NAMES, DCT_POSITION_PRESETS
from ui_flows.utils import generate_dummy_message_callback, reset_embed_state
# Ganti ini dengan impor DCT Anda yang sebenarnya
from methods.frequency.dct import DCTSteganography, DCT_DEFAULT_PARAM 
from metrics.impercability import SteganographyMetrics
from metrics.robustness import RobustnessTester, ATTACK_CONFIGURATIONS
from helpers.message_binary import message_to_binary

def draw_dct_embed_tab():
    """Menampilkan UI untuk tab Embed DCT."""
    
    with st.expander("Configure Parameters"):
        dct_block_size = st.number_input("Block Size", min_value=4, max_value=16, value=DCT_DEFAULT_PARAM['block_size'], step=4, key="dct_block_size",
                                         on_change=reset_embed_state, args=("dct",))
        dct_quant_factor = st.number_input("Quantization Factor", min_value=1, max_value=100, value=DCT_DEFAULT_PARAM['quant_factor'], key="dct_quant_factor",
                                           on_change=reset_embed_state, args=("dct",))
        dct_preset_name = st.selectbox("Embedding Position Preset", options=DCT_PRESET_NAMES, index=1, key="dct_embed_pos_preset",
                                       on_change=reset_embed_state, args=("dct",))
        dct_embed_positions = DCT_POSITION_PRESETS[dct_preset_name]

    st.subheader("Message Input")
    col1, col2 = st.columns([0.6, 0.4]) 
    with col1:
        dct_embed_msg = st.text_area("Your Secret Message", height=150, key="dct_embed_msg", label_visibility="collapsed",
                                     on_change=reset_embed_state, args=("dct",))
    with col2:
        with st.container(border=True):
            st.write("Dummy Message Helper")
            dummy_bit_len = st.number_input("Target Bit Length", min_value=8, value=DEFAULT_TARGET_BIT_SIZE, step=8, key="dct_dummy_bit_len")
            st.button("Generate Dummy Message", key="dct_dummy_btn", on_click=generate_dummy_message_callback, args=("dct_dummy_bit_len", "dct_embed_msg"))
    
    st.subheader("Cover Image")
    cover_image_file = st.file_uploader("Upload Cover Image", type=["png", "jpg", "bmp"], key="dct_embed_img", label_visibility="collapsed",
                                        on_change=reset_embed_state, args=("dct",))
        
    if cover_image_file is not None:
        st.image(cover_image_file, caption="Uploaded Cover Image", use_container_width=True)
        
    st.divider()
    embed_button = st.button("Embed Message", type="primary", key="dct_embed_btn")

    if embed_button:
        if cover_image_file is None:
            st.error("Please upload a cover image first!")
            st.session_state.dct_stego_image_bytes = None
            st.session_state.dct_params_json = None
        else:
            try:
                file_bytes = np.asarray(bytearray(cover_image_file.read()), dtype=np.uint8)
                cover_image_rgb = cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                
                # --- GANTI INI DENGAN LOGIKA DCT ANDA ---
                st.warning("Logika Embed DCT belum diimplementasikan di UI.")
                # dct_stego = DCTSteganography(block_size=dct_block_size, quant_factor=dct_quant_factor, embed_positions=dct_embed_positions)
                # stego_image_np, final_bit_length = dct_stego.embed(cover_image_rgb, dct_embed_msg)
                
                # Placeholder
                stego_image_np = cover_image_rgb 
                final_bit_length = len(dct_embed_msg.encode('utf-8')) * 8
                # --- AKHIR PLACEHOLDER ---

                stego_image_pil = Image.fromarray(stego_image_np)
                buffer = BytesIO()
                stego_image_pil.save(buffer, format="PNG")
                
                parameters_to_save = {
                    "method": METHOD_DCT, 
                    "block_size": dct_block_size,
                    "quant_factor": dct_quant_factor,
                    "embed_positions": dct_embed_positions,
                    "message_bit_length": final_bit_length
                }
                
                st.session_state.dct_stego_image_bytes = buffer.getvalue()
                st.session_state.dct_params_json = json.dumps(parameters_to_save, indent=4)
                        
            except Exception as e:
                st.error(f"Embedding Failed: {e}")
                st.session_state.dct_stego_image_bytes = None
                st.session_state.dct_params_json = None
    
    if st.session_state.dct_stego_image_bytes is not None:
        st.subheader("Embedding Results")
        # (Logika tampilan hasil... sama seperti LSB)
        st.image(st.session_state.dct_stego_image_bytes)
        st.download_button("Download Image", st.session_state.dct_stego_image_bytes, "dct_stego.png")
        st.download_button("Download Params", st.session_state.dct_params_json, "dct_params.json")


def draw_dct_extract_tab():
    """Menampilkan UI untuk tab Extract DCT."""
    st.info("DCT Extract UI belum diimplementasikan.")
    # (Salin logika dari lsb_ui.py dan sesuaikan untuk parameter DCT)