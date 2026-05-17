import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import tempfile
import os

# ================================
# 1. PAGE SETUP
# ================================
st.set_page_config(page_title="Audio Classifier", page_icon="🎤")
st.title("🎤 Audio Classification App")
st.write("Apni `.wav` audio file upload karein aur model batayega ke yeh **'no'**, **'stop'**, ya **'yes'** hai.")

# ================================
# 2. LOAD SAVED MODEL (Cached for speed)
# ================================
@st.cache_resource
def load_model():
    # Aapka GitHub Release wala model link
    model_url = "https://github.com/Rana-affi/audio-classification-app/releases/download/v1.0/audio_model.h5" 
    
    # Model download karna
    model_path = tf.keras.utils.get_file("audio_model.h5", origin=model_url)
    
    # Keras 3 ke 'pop from empty list' error ko khatam karne ke liye strict loading off ki hai
    return tf.keras.models.load_model(model_path, compile=False, safe_mode=False)

try:
    model = load_model()
except Exception as e:
    # Agar phir bhi masla kare, to direct legacy format se load karne ki koshish karna
    try:
        import keras
        model = keras.models.load_model(model_path, compile=False)
    except:
        st.error(f"Error loading model: {e}")
        st.stop()

# ================================
# 3. LABELS
# ================================
label_names = np.array(['no', 'stop', 'yes'])


# ================================
# 4. SPECTROGRAM FUNCTION
# ================================
def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(
        waveform,
        frame_length=255,
        frame_step=128
    )
    spectrogram = tf.abs(spectrogram)
    return spectrogram[..., tf.newaxis]


# ================================
# 5. PREDICTION FUNCTION
# ================================
def predict_audio(file_path):
    # load audio
    audio, sr = librosa.load(file_path, sr=16000)

    # convert to spectrogram
    x = get_spectrogram(audio)
    x = tf.expand_dims(x, axis=0)

    # prediction
    prediction = model(x)

    # result
    pred_index = np.argmax(prediction, axis=1)[0]
    return label_names[pred_index]


# ================================
# 6. FRONTEND UI
# ================================
st.markdown("---")
# User se file upload karwana
uploaded_file = st.file_uploader("Audio file upload karein (WAV format)", type=["wav"])

if uploaded_file is not None:
    # Audio play karne ka option
    st.audio(uploaded_file, format='audio/wav')

    # Predict button
    if st.button("🔍 Predict Audio"):
        with st.spinner("Prediction ho rahi hai..."):

            # File ko temporary save karna taake librosa usko read kar sake
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            try:
                # Prediction function ko call karna
                result = predict_audio(tmp_path)

                # Result screen par dikhana
                st.success(f"🎉 **Prediction:** Aapki audio mein **'{result.upper()}'** kaha gaya hai!")

            except Exception as e:
                st.error(f"Prediction mein error aagaya: {e}")

            finally:
                # Temporary file delete karna
                os.remove(tmp_path)
