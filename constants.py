# constants.py

# --- Helpers ---
DEFAULT_MESSAGE = (
    "This message uses deterministic padding. 🚀 "
    "The padding is a repeating, non-random sequence. "
    "Let's test it: 123!@#<>? éçñ. 😊👍"
)
DEFAULT_TARGET_BIT_SIZE = 512

# --- Method Names ---
METHOD_LSB = "Spatial - LSB"
METHOD_PVD = "Spatial - PVD"
METHOD_EMD = "Spatial - EMD"
METHOD_DCT = "Frequency - DCT"
METHOD_DWT = "Frequency - DWT"
METHOD_FFT = "Frequency - FFT"
METHOD_DCT_LSB = "Hybrid - DCT + LSB"
METHOD_DCT_PVD = "Hybrid - DCT + PVD"
METHOD_DCT_EMD = "Hybrid - DCT + EMD" # <--- DITAMBAHKAN

METHODS_ALL = (
    METHOD_LSB, METHOD_PVD, METHOD_EMD, 
    METHOD_DCT, METHOD_DWT, METHOD_FFT, 
    METHOD_DCT_LSB, METHOD_DCT_PVD, METHOD_DCT_EMD # <--- DIPERBARUI
)

# --- DCT Positions ---
DCT_POSITION_MID_LOW = [(1, 1), (2, 0), (0, 2), (3, 0), (0, 3)]
DCT_POSITION_MID = [(2, 1), (1, 2), (2, 2), (3, 1), (1, 3)]
DCT_POSITION_HIGH = [(4, 4), (5, 3), (3, 5), (6, 2), (2, 6)]

DCT_POSITION_PRESETS = {
    "Mid-Low Frequencies": DCT_POSITION_MID_LOW,
    "Mid Frequencies": DCT_POSITION_MID,
    "High Frequencies": DCT_POSITION_HIGH
}
DCT_PRESET_NAMES = list(DCT_POSITION_PRESETS.keys())