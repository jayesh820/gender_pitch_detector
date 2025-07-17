import streamlit as st
import librosa
import numpy as np
import tempfile
import os
import time
import matplotlib.pyplot as plt
import random

# Initialize session state for entry
if "start_app" not in st.session_state:
    st.session_state.start_app = False
if "agreed" not in st.session_state:
    st.session_state.agreed = False

# Splash Screen (Welcome + Flip Style Transition)
if not st.session_state.start_app:
    st.markdown("""
        <style>
        body {
            background-color: #000000;
        }
        .splash {
            text-align: center;
            margin-top: 100px;
            animation: fadein 2s;
        }
        .splash h1 {
            font-size: 48px;
            color: #00ffe0;
            animation: glow 2s ease-in-out infinite alternate;
        }
        .splash button {
            font-size: 20px;
            padding: 12px 25px;
            background-color: #00bcd4;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            margin-top: 30px;
        }
        @keyframes glow {
            from { text-shadow: 0 0 10px #00ffe0; }
            to { text-shadow: 0 0 20px #00c2ff; }
        }
        </style>
        <div class="splash">
            <h1>🎧 Voice Analyzer AI</h1>
            <p style="color:white">Welcome! Analyze your voice using smart AI.</p>
        </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Start Analysis"):
        st.session_state.start_app = True
        st.rerun()

# Disclaimer Page (User Consent)
elif not st.session_state.agreed:
    st.markdown("""
        <div style='text-align: center; margin-top: 80px;'>
            <h2 style='color: #f39c12;'>⚠️ Voice Privacy Disclaimer</h2>
            <p style='color: white;'>This app uses AI to analyze your voice and predict features like gender, pitch, and energy.</p>
            <p style='color: white;'>No data is stored or shared.</p>
        </div>
    """, unsafe_allow_html=True)
    if st.button("✅ I Agree & Continue"):
        st.session_state.agreed = True
        st.rerun()

# Main App Starts After Entry Screens
else:
    st.markdown("""
        <style>
        html, body, .stApp {
            background-image: url('https://cdn.pixabay.com/photo/2020/04/22/19/02/sound-5077899_1280.jpg');
            background-size: cover;
            background-attachment: fixed;
        }
        .app-box {
            background-color: #fff9db;
            padding: 30px 40px;
            border-radius: 20px;
            max-width: 850px;
            margin: 40px auto;
            box-shadow: 0px 8px 24px rgba(0,0,0,0.2);
        }
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 15px;
            margin-top: 30px;
        }
        .feature-box {
            padding: 15px;
            border-radius: 12px;
            color: white;
            font-weight: bold;
            font-size: 16px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
        }
        </style>
    """, unsafe_allow_html=True)

    def extract_voice_features(audio_path):
        y, sr = librosa.load(audio_path, sr=None)
        pitches = librosa.yin(y, fmin=75, fmax=300, sr=sr)
        avg_pitch = np.mean(pitches)
        gender = "Female 👩" if avg_pitch > 160 else "Male 👨"
        duration = librosa.get_duration(y=y, sr=sr)
        energy = np.mean(np.square(y))
        rms = np.mean(librosa.feature.rms(y=y)[0])
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y)[0])
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0])
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0])
        return {
            "gender": gender,
            "pitch": avg_pitch,
            "duration": duration,
            "energy": energy,
            "rms": rms,
            "zcr": zcr,
            "centroid": centroid,
            "bandwidth": bandwidth,
            "rolloff": rolloff,
            "pitch_array": pitches
        }

    st.markdown('<div class="app-box">', unsafe_allow_html=True)
    st.title("🎧 AI Gender & Feature Detector")
    st.write("Upload a voice file to detect gender and view features")

    audio_data = None
    uploaded_file = st.file_uploader("📤 Upload voice file (WAV/MP3)", type=["wav", "mp3"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            audio_data = tmp.name
        st.audio(audio_data)

    if audio_data:
        with st.spinner("⏳ Analyzing your voice..."):
            time.sleep(1)
            try:
                features = extract_voice_features(audio_data)

                st.markdown(f"""
                    <div style='padding: 20px; background-color: #1abc9c; color: white;
                                border-radius: 10px; text-align: center; font-size: 22px; margin-top: 20px;'>
                        👤 Detected Gender: <b>{features['gender']}</b>
                    </div>
                """, unsafe_allow_html=True)

                colors = ["#e74c3c", "#3498db", "#9b59b6", "#f1c40f", "#1abc9c", "#34495e", "#e67e22", "#2ecc71"]
                labels = [
                    ("🎤 Avg Pitch", f"{features['pitch']:.2f} Hz"),
                    ("⏱ Duration", f"{features['duration']:.2f} sec"),
                    ("⚡ Energy", f"{features['energy']:.4f}"),
                    ("🔊 RMS", f"{features['rms']:.4f}"),
                    ("🎚 ZCR", f"{features['zcr']:.4f}"),
                    ("📈 Centroid", f"{features['centroid']:.2f}"),
                    ("📉 Bandwidth", f"{features['bandwidth']:.2f}"),
                    ("📦 Rolloff", f"{features['rolloff']:.2f}")
                ]

                st.markdown('<div class="feature-grid">', unsafe_allow_html=True)
                for i, (title, value) in enumerate(labels):
                    color = colors[i % len(colors)]
                    st.markdown(f"""
                        <div class='feature-box' style='background-color: {color};'>
                            {title}: {value}
                        </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                with st.expander("📊 View Pitch Over Time"):
                    fig, ax = plt.subplots()
                    ax.plot(features["pitch_array"], color="orange")
                    ax.set_title("Pitch Over Time")
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
            os.unlink(audio_data)
    st.markdown('</div>', unsafe_allow_html=True)
