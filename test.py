from src.styletts2 import tts

model = tts.StyleTTS2(
    phoneme_converter="espeak",  # "espeak" > "gruut"
    local="models/",  # where cached_path will store downloaded files
    device="cuda",  # "cuda" > "cpu"
    debug=False,
)

ref_s = model.compute_style("voices/m-us-2.wav")  # filepath to .wav file

out = model.inference(
    "Hello there, I am now a python package.",
    output_wav_file="test2.wav",
    alpha=0.3,
    beta=0.3,
    diffusion_steps=10,
    embedding_scale=1,
    speed=1.4,
    ref_s=ref_s,
)
