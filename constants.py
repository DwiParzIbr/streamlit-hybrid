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
METHOD_EMD = "Spatial - EMD" # <--- DITAMBAHKAN
METHOD_DCT = "Frequency - DCT"
METHOD_DWT = "Frequency - DWT" # <--- DITAMBAHKAN
# METHOD_HYBRID = "Hybrid - DCT + LSB" # Dihapus
METHODS_ALL = (METHOD_LSB, METHOD_PVD, METHOD_EMD, METHOD_DCT, METHOD_DWT) # <--- DIPERBARUI

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