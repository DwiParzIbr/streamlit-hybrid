import streamlit as st
import json
import numpy as np
import pandas as pd
from methods.frequency.dct import DCT_POSITION_MID, DCT_POSITION_MID_LOW, DCT_POSITION_HIGH

# Kamus untuk memetakan nama preset ke data list
DCT_POSITION_PRESETS = {
    "Mid-Low Frequencies": DCT_POSITION_MID_LOW,
    "Mid Frequencies": DCT_POSITION_MID,
    "High Frequencies": DCT_POSITION_HIGH
}
# Daftar nama preset untuk selectbox
DCT_PRESET_NAMES = list(DCT_POSITION_PRESETS.keys())


# --- Sidebar Navigation ---
st.sidebar.title("Hybrid Steganography 🛡️")
st.sidebar.write("Select your steganography technique:")

selected_method = st.sidebar.radio(
    "Choose your method:",
    ("Spatial - LSB", "Frequency - DCT", "Hybrid - DCT + LSB"),
    label_visibility="collapsed"
)

# --- Main Content Area ---
tab_embed, tab_extract = st.tabs(["Embed Message 🖼️", "Extract Message 🕵️"])

# --- Embed Tab Content ---
with tab_embed:
    st.title(f"Embed using {selected_method}")
    
    # --- FLOW LSB ---
    if selected_method == "Spatial - LSB":
        
        with st.expander("Configure Parameters"):
            lsb_bit_plane = st.number_input("LSB Bit Plane (e.g., 1-8)", min_value=1, max_value=8, value=1, key="lsb_bit_plane")

        st.subheader("Message Input")
        col1, col2 = st.columns([0.6, 0.4]) 

        with col1:
            lsb_embed_msg = st.text_area("Your Secret Message", height=150, key="lsb_embed_msg", label_visibility="collapsed")
        
        with col2:
            with st.container(border=True):
                st.write("Dummy Message Helper")
                dummy_len = st.number_input("Message Length", min_value=10, value=100)
                if st.button("Generate Dummy Message", key="lsb_dummy_btn"):
                    st.toast(f"Dummy message of length {dummy_len} generated!")
        
        st.subheader("Cover Image")
        cover_image_file = st.file_uploader("Upload Cover Image", type=["png", "jpg", "bmp"], key="lsb_embed_img", label_visibility="collapsed")
            
        if cover_image_file is not None:
            st.image(cover_image_file, caption="Uploaded Cover Image", use_container_width=True)
            
        st.divider()
        embed_button = st.button("Embed Message", type="primary", key="lsb_embed_btn")

        if embed_button:
            st.subheader("Embedding Results")
            if cover_image_file is None:
                st.error("Please upload a cover image first!")
            else:
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.write("**Stego-Image**")
                    st.image("https://via.placeholder.com/400x400.png?text=Your+Stego-Image", caption="Steganographic Image", use_container_width=True)
                    st.download_button(
                        label="Download Image",
                        data=b"placeholder_image_data", 
                        file_name="stego_image.png",
                        mime="image/png"
                    )
                
                with res_col2:
                    st.write("**Parameters Used**")
                    message_bit_length = len(lsb_embed_msg.encode('utf-8')) * 8
                    parameters_to_save = {
                            "method": "LSB", 
                            "bit_plane": lsb_bit_plane,
                            "message_bit_length": message_bit_length
                    }
                    # FIX: Konversi dict ke string JSON untuk di-download
                    parameters_json = json.dumps(parameters_to_save, indent=4)
                    
                    st.download_button(
                        label="Download Parameters",
                        data=parameters_json, # Menggunakan string JSON
                        file_name="parameters.json",
                        mime="application/json"
                    )
    
    # --- FLOW DCT (Diperbarui) ---
    elif selected_method == "Frequency - DCT":
        
        with st.expander("Configure Parameters"):
            dct_block_size = st.number_input("Block Size", min_value=4, max_value=16, value=8, step=4, key="dct_block_size")
            dct_quant_factor = st.number_input("Quantization Factor", min_value=1, max_value=100, value=70, key="dct_quant_factor")
            
            # --- PERUBAHAN DI SINI: multiselect -> selectbox ---
            dct_preset_name = st.selectbox(
                "Embedding Position Preset", 
                options=DCT_PRESET_NAMES, 
                index=1, # Default ke "Mid Frequencies"
                key="dct_embed_pos_preset"
            )
            # Dapatkan list posisi aktual dari preset yang dipilih
            dct_embed_positions = DCT_POSITION_PRESETS[dct_preset_name]
            # --- AKHIR PERUBAHAN ---

        st.subheader("Message Input")
        col1, col2 = st.columns([0.6, 0.4]) 

        with col1:
            dct_embed_msg = st.text_area("Your Secret Message", height=150, key="dct_embed_msg", label_visibility="collapsed")
        
        with col2:
            with st.container(border=True):
                st.write("Dummy Message Helper")
                dummy_len = st.number_input("Message Length", min_value=10, value=100, key="dct_dummy_len")
                if st.button("Generate Dummy Message", key="dct_dummy_btn"):
                    st.toast(f"Dummy message of length {dummy_len} generated!")
        
        st.subheader("Cover Image")
        cover_image_file = st.file_uploader("Upload Cover Image", type=["png", "jpg", "bmp"], key="dct_embed_img", label_visibility="collapsed")
            
        if cover_image_file is not None:
            st.image(cover_image_file, caption="Uploaded Cover Image", use_container_width=True)
            
        st.divider()
        embed_button = st.button("Embed Message", type="primary", key="dct_embed_btn")

        if embed_button:
            st.subheader("Embedding Results")
            if cover_image_file is None:
                st.error("Please upload a cover image first!")
            else:
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.write("**Stego-Image**")
                    st.image("https://via.placeholder.com/400x400.png?text=Your+Stego-Image", caption="Steganographic Image", use_container_width=True)
                    st.download_button(
                        label="Download Image",
                        data=b"placeholder_image_data", 
                        file_name="stego_image.png",
                        mime="image/png"
                    )
                
                with res_col2:
                    st.write("**Parameters Used**")
                    message_bit_length = len(dct_embed_msg.encode('utf-8')) * 8
                    parameters_to_save = {
                            "method": "DCT", 
                            "block_size": dct_block_size,
                            "quant_factor": dct_quant_factor,
                            "embed_positions": dct_embed_positions, # Simpan list aktual
                            "message_bit_length": message_bit_length
                    }
                    parameters_json = json.dumps(parameters_to_save, indent=4)
                    
                    st.download_button(
                        label="Download Parameters",
                        data=parameters_json,
                        file_name="parameters.json",
                        mime="application/json"
                    )

    # --- FLOW HYBRID (Diperbarui) ---
    elif selected_method == "Hybrid - DCT + LSB":
        
        with st.expander("Configure Parameters"):
            hybrid_dct_ratio = st.slider("DCT Ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.1, key="hybrid_ratio")
            st.write(f"**DCT Ratio: {hybrid_dct_ratio*100:.0f}%** | **LSB Ratio: {(1-hybrid_dct_ratio)*100:.0f}%**")
            
            with st.container(border=True):
                st.write("**DCT Parameters**")
                hybrid_quant_factor = st.number_input("Quantization Factor", min_value=1, max_value=100, value=70, key="hybrid_quant_factor")
                
                # --- PERUBAHAN DI SINI: multiselect -> selectbox ---
                hybrid_preset_name = st.selectbox(
                    "Embedding Position Preset", 
                    options=DCT_PRESET_NAMES, 
                    index=1, # Default ke "Mid Frequencies"
                    key="hybrid_embed_pos_preset"
                )
                hybrid_embed_positions = DCT_POSITION_PRESETS[hybrid_preset_name]
                # --- AKHIR PERUBAHAN ---
            
            with st.container(border=True):
                st.write("**LSB Parameters**")
                hybrid_lsb_bits = st.number_input("LSB Bit Plane (e.g., 1-8)", min_value=1, max_value=8, value=1, key="hybrid_lsb_bits")

        st.subheader("Message Input")
        col1, col2 = st.columns([0.6, 0.4]) 

        with col1:
            hybrid_embed_msg = st.text_area("Your Secret Message", height=150, key="hybrid_embed_msg", label_visibility="collapsed")
        
        with col2:
            with st.container(border=True):
                st.write("Dummy Message Helper")
                dummy_len = st.number_input("Message Length", min_value=10, value=100, key="hybrid_dummy_len")
                if st.button("Generate Dummy Message", key="hybrid_dummy_btn"):
                    st.toast(f"Dummy message of length {dummy_len} generated!")
        
        st.subheader("Cover Image")
        cover_image_file = st.file_uploader("Upload Cover Image", type=["png", "jpg", "bmp"], key="hybrid_embed_img", label_visibility="collapsed")
            
        if cover_image_file is not None:
            st.image(cover_image_file, caption="Uploaded Cover Image", use_container_width=True)
            
        st.divider()
        embed_button = st.button("Embed Message", type="primary", key="hybrid_embed_btn")

        if embed_button:
            st.subheader("Embedding Results")
            if cover_image_file is None:
                st.error("Please upload a cover image first!")
            else:
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.write("**Stego-Image**")
                    st.image("https://via.placeholder.com/400x400.png?text=Your+Stego-Image", caption="Steganographic Image", use_container_width=True)
                    st.download_button(
                        label="Download Image",
                        data=b"placeholder_image_data", 
                        file_name="stego_image.png",
                        mime="image/png"
                    )
                
                with res_col2:
                    st.write("**Parameters Used**")
                    message_bit_length = len(hybrid_embed_msg.encode('utf-8')) * 8
                    parameters_to_save = {
                        "method": "Hybrid (DCT + LSB)",
                        "dct_lsb_ratio": (hybrid_dct_ratio, 1.0 - hybrid_dct_ratio),
                        "dct_params": {
                            "quant_factor": hybrid_quant_factor, 
                            "embed_positions": hybrid_embed_positions # Simpan list aktual
                        },
                        "lsb_params": {
                            "bits_per_channel": hybrid_lsb_bits
                        },
                        "message_bit_length": message_bit_length
                    }
                    parameters_json = json.dumps(parameters_to_save, indent=4)
                    
                    st.download_button(
                        label="Download Parameters",
                        data=parameters_json,
                        file_name="parameters.json",
                        mime="application/json"
                    )

# --- Extract Tab Content ---
with tab_extract:
    st.title(f"Extract using {selected_method}")
    
    # --- FLOW LSB (Tidak Berubah) ---
    if selected_method == "Spatial - LSB":
        
        with st.expander("Configure Parameters"):
            lsb_bit_plane_extract = st.number_input("LSB Bit Plane (e.g., 1-8)", min_value=1, max_value=8, value=1, key="lsb_bit_plane_extract")

        st.subheader("Upload Stego-Image")
        stego_image_file = st.file_uploader("Upload the image you want to extract from", type=["png", "jpg", "bmp"], key="lsb_extract_img", label_visibility="collapsed")

        if stego_image_file is not None:
            st.image(stego_image_file, caption="Uploaded Stego-Image", use_container_width=True)

        st.subheader("Message Bit Length")
        message_bit_length_extract = st.number_input(
            "Enter the bit length of the secret message", 
            min_value=1, value=1024, key="lsb_extract_bit_length"
        )
        
        st.subheader("Optional Inputs for Metrics")
        col1, col2 = st.columns(2)
        with col1:
            optional_cover_image = st.file_uploader("Original Cover Image (for Imperceptibility)", type=["png", "jpg", "bmp"], key="lsb_extract_cover")
        with col2:
            optional_original_message = st.text_area("Original Message (for Robustness)", height=155, key="lsb_extract_msg")

        st.divider()
        extract_button = st.button("Extract Message", type="primary", key="lsb_extract_btn")

        if extract_button:
            # (Logika hasil LSB tetap sama)
            st.write("...Hasil Ekstraksi LSB...")

    # --- FLOW DCT (Diperbarui) ---
    elif selected_method == "Frequency - DCT":
        
        with st.expander("Configure Parameters"):
            dct_block_size_ext = st.number_input("Block Size", min_value=4, max_value=16, value=8, step=4, key="dct_ext_block_size")
            dct_quant_factor_ext = st.number_input("Quantization Factor", min_value=1, max_value=100, value=70, key="dct_ext_quant_factor")
            
            # --- PERUBAHAN DI SINI: multiselect -> selectbox ---
            dct_preset_name_ext = st.selectbox(
                "Embedding Position Preset", 
                options=DCT_PRESET_NAMES, 
                index=1, # Default ke "Mid Frequencies"
                key="dct_ext_embed_pos_preset"
            )
            dct_embed_positions_ext = DCT_POSITION_PRESETS[dct_preset_name_ext]
            # --- AKHIR PERUBAHAN ---


        st.subheader("Upload Stego-Image")
        stego_image_file = st.file_uploader("Upload the image you want to extract from", type=["png", "jpg", "bmp"], key="dct_extract_img", label_visibility="collapsed")

        if stego_image_file is not None:
            st.image(stego_image_file, caption="Uploaded Stego-Image", use_container_width=True)

        st.subheader("Message Bit Length")
        message_bit_length_extract = st.number_input(
            "Enter the bit length of the secret message", 
            min_value=1, value=1024, key="dct_extract_bit_length"
        )
        
        st.subheader("Optional Inputs for Metrics")
        col1, col2 = st.columns(2)
        with col1:
            optional_cover_image = st.file_uploader("Original Cover Image (for Imperceptibility)", type=["png", "jpg", "bmp"], key="dct_extract_cover")
        with col2:
            optional_original_message = st.text_area("Original Message (for Robustness)", height=155, key="dct_extract_msg")

        st.divider()
        extract_button = st.button("Extract Message", type="primary", key="dct_extract_btn")

        if extract_button:
            # (Logika hasil DCT akan ditempatkan di sini)
            st.write("...Hasil Ekstraksi DCT...")

    # --- FLOW HYBRID (Diperbarui) ---
    elif selected_method == "Hybrid - DCT + LSB":
        
        with st.expander("Configure Parameters"):
            hybrid_dct_ratio_ext = st.slider("DCT Ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.1, key="hybrid_ext_ratio")
            st.write(f"**DCT Ratio: {hybrid_dct_ratio_ext*100:.0f}%** | **LSB Ratio: {(1-hybrid_dct_ratio_ext)*100:.0f}%**")
            
            with st.container(border=True):
                st.write("**DCT Parameters**")
                hybrid_quant_factor_ext = st.number_input("Quantization Factor", min_value=1, max_value=100, value=70, key="hybrid_ext_quant_factor")
                
                # --- PERUBAHAN DI SINI: multiselect -> selectbox ---
                hybrid_preset_name_ext = st.selectbox(
                    "Embedding Position Preset", 
                    options=DCT_PRESET_NAMES, 
                    index=1, # Default ke "Mid Frequencies"
                    key="hybrid_ext_embed_pos_preset"
                )
                hybrid_embed_positions_ext = DCT_POSITION_PRESETS[hybrid_preset_name_ext]
                # --- AKHIR PERUBAHAN ---
            
            with st.container(border=True):
                st.write("**LSB Parameters**")
                hybrid_lsb_bits_ext = st.number_input("LSB Bit Plane (e.g., 1-8)", min_value=1, max_value=8, value=1, key="hybrid_ext_lsb_bits")

        st.subheader("Upload Stego-Image")
        stego_image_file = st.file_uploader("Upload the image you want to extract from", type=["png", "jpg", "bmp"], key="hybrid_extract_img", label_visibility="collapsed")

        if stego_image_file is not None:
            st.image(stego_image_file, caption="Uploaded Stego-Image", use_container_width=True)

        st.subheader("Message Bit Length")
        message_bit_length_extract = st.number_input(
            "Enter the bit length of the secret message", 
            min_value=1, value=1024, key="hybrid_extract_bit_length"
        )
        
        st.subheader("Optional Inputs for Metrics")
        col1, col2 = st.columns(2)
        with col1:
            optional_cover_image = st.file_uploader("Original Cover Image (for Imperceptibility)", type=["png", "jpg", "bmp"], key="hybrid_extract_cover")
        with col2:
            optional_original_message = st.text_area("Original Message (for Robustness)", height=155, key="hybrid_extract_msg")

        st.divider()
        extract_button = st.button("Extract Message", type="primary", key="hybrid_extract_btn")

        if extract_button:
            # (Logika hasil Hybrid akan ditempatkan di sini)
            st.write("...Hasil Ekstraksi Hybrid...")