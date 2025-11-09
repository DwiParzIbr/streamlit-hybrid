import streamlit as st
import json
import numpy as np
import pandas as pd
import cv2 
from PIL import Image
from io import BytesIO

from helpers.message_generator import create_pattern_padded_message, DEFAULT_MESSAGE, DEFAULT_TARGET_BIT_SIZE
from methods.spatial.lsb import LSBSteganography, LSB_DEFAULT_PARAM
from methods.frequency.dct import DCT_POSITION_MID_LOW, DCT_POSITION_MID, DCT_POSITION_HIGH
from metrics.impercability import SteganographyMetrics
from metrics.robustness import RobustnessTester, ATTACK_CONFIGURATIONS
from helpers.message_binary import message_to_binary, binary_to_message

def generate_dummy_message_callback(target_bit_len_key, message_key):
    """
    Fungsi callback yang aman untuk memodifikasi st.session_state 
    yang terikat pada st.text_area.
    """
    dummy_bit_len = st.session_state[target_bit_len_key]
    
    target_byte_size = (dummy_bit_len + 7) // 8 
    
    padded_msg = create_pattern_padded_message(DEFAULT_MESSAGE, target_byte_size)
    
    st.session_state[message_key] = padded_msg
    
    # --- PERBAIKAN KECIL PADA TOAST ---
    st.toast(f"Message padded to target {target_byte_size} bytes ({dummy_bit_len} bits)!")

def reset_embed_state(method_key_prefix):
    """Menghapus hasil yang disimpan di session_state saat input berubah."""
    image_key = f"{method_key_prefix}_stego_image_bytes"
    params_key = f"{method_key_prefix}_params_json"
    
    if image_key in st.session_state:
        st.session_state[image_key] = None
    if params_key in st.session_state:
        st.session_state[params_key] = None
        
# --- CONSTANTS ---
METHOD_LSB = "Spatial - LSB"
METHOD_DCT = "Frequency - DCT"
METHOD_HYBRID = "Hybrid - DCT + LSB"
METHODS_ALL = (METHOD_LSB, METHOD_DCT, METHOD_HYBRID)

DCT_POSITION_PRESETS = {
    "Mid-Low Frequencies": DCT_POSITION_MID_LOW,
    "Mid Frequencies": DCT_POSITION_MID,
    "High Frequencies": DCT_POSITION_HIGH
}
DCT_PRESET_NAMES = list(DCT_POSITION_PRESETS.keys())


# --- Session State Initialization ---
default_byte_size = (DEFAULT_TARGET_BIT_SIZE + 7) // 8
default_padded_msg = create_pattern_padded_message(DEFAULT_MESSAGE, default_byte_size)

if 'lsb_embed_msg' not in st.session_state: st.session_state.lsb_embed_msg = default_padded_msg
if 'dct_embed_msg' not in st.session_state: st.session_state.dct_embed_msg = default_padded_msg
if 'hybrid_embed_msg' not in st.session_state: st.session_state.hybrid_embed_msg = default_padded_msg

if 'lsb_stego_image_bytes' not in st.session_state: st.session_state.lsb_stego_image_bytes = None
if 'lsb_params_json' not in st.session_state: st.session_state.lsb_params_json = None
if 'dct_stego_image_bytes' not in st.session_state: st.session_state.dct_stego_image_bytes = None
if 'dct_params_json' not in st.session_state: st.session_state.dct_params_json = None
if 'hybrid_stego_image_bytes' not in st.session_state: st.session_state.hybrid_stego_image_bytes = None
if 'hybrid_params_json' not in st.session_state: st.session_state.hybrid_params_json = None

# --- PERUBAHAN DI SINI: Inisialisasi state Ekstraksi ---
# Menggunakan nama key yang konsisten (bits_per_channel)
if 'lsb_extract_bits_per_channel' not in st.session_state:
    st.session_state.lsb_extract_bits_per_channel = LSB_DEFAULT_PARAM['bits_per_channel']
if 'lsb_extract_bit_length' not in st.session_state:
    st.session_state.lsb_extract_bit_length = 512
# --- AKHIR PERUBAHAN ---


# --- Sidebar Navigation ---
st.sidebar.title("Hybrid Steganography 🛡️")
st.sidebar.header("Spatial Domain")
st.sidebar.write(f"• {METHOD_LSB}")
st.sidebar.divider()
st.sidebar.header("Frequency Domain")
st.sidebar.write(f"• {METHOD_DCT}")
st.sidebar.divider()
st.sidebar.header("Hybrid")
st.sidebar.write(f"• {METHOD_HYBRID}")
st.sidebar.divider()
selected_method = st.sidebar.radio("Choose Method", METHODS_ALL, key="final_method_selector", label_visibility="collapsed")

tab_embed, tab_extract = st.tabs(["Embed Message 🖼️", "Extract Message 🕵️"])

# --- Embed Tab Content ---
with tab_embed:
    st.title(f"Embed using {selected_method}")
    
    if selected_method == METHOD_LSB:
        
        with st.expander("Configure Parameters"):
            # --- PERUBAHAN DI SINI: Label dan Key konsisten ---
            bits_per_channel_embed = st.number_input(
                "Bits Per Channel (e.g., 1-8)", min_value=1, max_value=8, 
                value=LSB_DEFAULT_PARAM['bits_per_channel'], 
                key="lsb_embed_bits_per_channel", # Menggunakan nama lengkap
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
                try:
                    file_bytes = np.asarray(bytearray(cover_image_file.read()), dtype=np.uint8)
                    cover_image_rgb = cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                    
                    lsb_stego = LSBSteganography(bits_per_channel=bits_per_channel_embed)
                    stego_image_np, final_bit_length = lsb_stego.embed(cover_image_rgb, lsb_embed_msg)

                    stego_image_pil = Image.fromarray(stego_image_np)
                    buffer = BytesIO()
                    stego_image_pil.save(buffer, format="PNG")
                    
                    # --- PERUBAHAN DI SINI: Key JSON konsisten ---
                    parameters_to_save = {
                        "method": METHOD_LSB, 
                        "bits_per_channel": bits_per_channel_embed, # Diubah dari bit_plane
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

# --- Extract Tab Content (FIXED) ---
with tab_extract:
    st.title(f"Extract using {selected_method}")
    
    if selected_method == METHOD_LSB:
        
        # 1. Load Parameters
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
                
                # --- PERUBAHAN DI SINI: Membaca key yang konsisten ---
                loaded_bits = param_data.get('bits_per_channel') # Diubah dari bit_plane
                loaded_bit_length = param_data.get('message_bit_length')

                if loaded_bits is not None:
                    # Mengatur state untuk key yang konsisten
                    st.session_state.lsb_extract_bits_per_channel = loaded_bits
                
                if loaded_bit_length is not None:
                    st.session_state.lsb_extract_bit_length = loaded_bit_length

                st.toast("Parameters loaded successfully!")

            except Exception as e:
                st.error(f"Failed to load parameters: {e}")

        # 2. Configure Parameters
        with st.expander("Configure Parameters"):
            # --- PERUBAHAN DI SINI: Menghapus 'value=' dan memperbarui key/label ---
            bits_per_channel_extract = st.number_input(
                "Bits Per Channel (e.g., 1-8)", 
                min_value=1, max_value=8, 
                key="lsb_extract_bits_per_channel" # Key konsisten
            )

        # 3. Upload Stego-Image
        st.subheader("3. Upload Stego-Image")
        stego_image_file_extract = st.file_uploader(
            "Upload the image you want to extract from", 
            type=["png", "jpg", "bmp"], 
            key="lsb_extract_img", 
            label_visibility="collapsed"
        )
        if stego_image_file_extract is not None:
            st.image(stego_image_file_extract, caption="Uploaded Stego-Image", use_container_width=True)

        # 4. Message Bit Length
        st.subheader("Message Bit Length")
        # --- PERUBAHAN DI SINI: Menghapus 'value=' ---
        message_bit_length_extract = st.number_input(
            "Enter the bit length of the secret message", 
            min_value=1, 
            key="lsb_extract_bit_length" # Key sudah konsisten
        )
        
        # 5. Optional Inputs for Metrics
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

        # 6. Extract Button
        st.divider()
        extract_button = st.button("Extract Message", type="primary", key="lsb_extract_btn")

        if extract_button:
            if stego_image_file_extract is None:
                st.error("Please upload a stego-image first!")
            else:
                try:
                    # --- 1. Logika Ekstraksi ---
                    
                    file_bytes = np.asarray(bytearray(stego_image_file_extract.read()), dtype=np.uint8)
                    stego_image_rgb = cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                    
                    lsb_stego_extract = LSBSteganography(bits_per_channel=bits_per_channel_extract)
                    extracted_message = lsb_stego_extract.extract(stego_image_rgb, message_bit_length_extract)
                    
                    st.subheader("Extracted Message")
                    st.text_area("Result", value=extracted_message, height=100, disabled=True)
                    
                    # --- 2. Metrik Imperceptibility (FORMAT DIPERBAIKI) ---
                    st.divider()
                    st.subheader("Imperceptibility Metrics")
                    if optional_cover_image is not None:
                        with st.spinner("Calculating imperceptibility metrics..."):
                            cover_file_bytes = np.asarray(bytearray(optional_cover_image.read()), dtype=np.uint8)
                            cover_image_rgb = cv2.cvtColor(cv2.imdecode(cover_file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                            
                            metrics = SteganographyMetrics(cover_image_rgb, stego_image_rgb)
                            all_metrics = metrics.get_all_metrics()
                            
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            # --- PERBAIKAN FORMATTING DI SINI ---
                            metric_col1.metric(label="MSE", value=f"{all_metrics['mse']:.4e}")
                            metric_col2.metric(label="PSNR (dB)", value=f"{all_metrics['psnr']:.4f}")
                            metric_col3.metric(label="SSIM", value=f"{all_metrics['ssim']:.8f}")
                    else:
                        st.info("Upload the original cover image to calculate imperceptibility metrics.")
                    
                    # --- 3. Metrik Robustness (DENGAN TAMPILAN GAMBAR) ---
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
                            
                            # --- PERBAIKAN TAMPILAN HASIL ROBUSTNESS ---
                            ber_data = []
                            # Kita perlu menyimpan gambar dan label untuk ditampilkan nanti
                            image_results = []
                            
                            for attack_label, (ber, attacked_img) in results.items():
                                ber_data.append({'Attack': attack_label, 'BER': ber})
                                if attacked_img is not None:
                                    image_results.append((attack_label, ber, attacked_img))
                            
                            df_ber = pd.DataFrame(ber_data)
                            st.dataframe(df_ber, height=300)

                            # Tampilkan grid gambar hasil serangan
                            st.subheader("Attacked Images")
                            
                            # Buat grid 4 kolom
                            cols = st.columns(4)
                            col_index = 0
                            
                            for label, ber, img in image_results:
                                with cols[col_index % 4]:
                                    st.image(img, caption=f"{label} (BER: {ber:.4f})", use_container_width=True)
                                col_index += 1
                            # --- AKHIR PERBAIKAN ---

                    else:
                        st.info("Enter the original message to calculate robustness (BER).")

                except Exception as e:
                    st.error(f"Extraction or Metrics Failed: {e}")

    # (Sisa flow DCT dan Hybrid dihilangkan untuk brevity)
    elif selected_method == METHOD_DCT:
        st.write("DCT Extract UI...")
    
    elif selected_method == METHOD_HYBRID:
        st.write("Hybrid Extract UI...")