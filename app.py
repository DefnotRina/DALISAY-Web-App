import streamlit as st
import cv2
import av
import numpy as np
import tempfile
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from PIL import Image

# 1. PAGE CONFIG
st.set_page_config(page_title="Marine Waste Detector", layout="centered", page_icon="🌊")

# 2. CSS: CLEAN UI
st.markdown("""
    <style>
        /* Hide Webrtc controls */
        div[data-testid="stWebcStreamer"] button { display: none !important; }
        div[data-testid="stWebcStreamer"] select { display: none !important; }
        div[data-testid="stWebcStreamer"] label { display: none !important; }
        div[data-testid="stWebcStreamer"] > div:nth-child(2) { display: none !important; }
        
        /* Align Title */
        h1 { text-align: center; }
        
        /* Big Buttons */
        div.stButton > button {
            width: 100%;
            font-size: 24px;
            padding: 8px;
            border-radius: 10px;
        }
        
        /* Style the Uploader to look like a big drop zone */
        div[data-testid="stFileUploader"] {
            padding-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# 3. LOAD MODEL
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 4. SESSION STATE
if "facing_mode" not in st.session_state:
    st.session_state.facing_mode = "environment" # Back Camera
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False 

# 5. HEADER
st.title("🌊 Marine Waste Detector")

# ==========================================
# 6. LIVE SCANNER (Real-time View)
# ==========================================
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    # Live view uses 320px for speed
    results = model(img, conf=0.25, imgsz=320)[0]
    annotated_frame = results.plot()
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# The Live Camera Feed
webrtc_streamer(
    key=f"waste-detection-{st.session_state.facing_mode}", 
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_configuration,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {"facingMode": st.session_state.facing_mode},
        "audio": False
    },
    async_processing=True,
)

# ==========================================
# 7. CONTROL BUTTONS (Aligned to Right)
# ==========================================
# [Spacer (Left)] | [Flip Button] | [Upload Button]
c_spacer, c_flip, c_upload = st.columns([4, 1, 1])

with c_flip:
    # 🔄 FLIP CAMERA
    if st.button("🔄", help="Switch Camera"):
        if st.session_state.facing_mode == "environment":
            st.session_state.facing_mode = "user"
        else:
            st.session_state.facing_mode = "environment"
        st.rerun()

with c_upload:
    # 📤 UNIVERSAL CAPTURE TOGGLE
    if st.button("📤", help="Capture Photo or Video"):
        st.session_state.show_uploader = not st.session_state.show_uploader
        st.rerun()

# ==========================================
# 8. THE UNIVERSAL DRAWER (Photo & Video)
# ==========================================
if st.session_state.show_uploader:
    st.markdown("---")
    st.write("### 📸 Capture or Upload")
    st.caption("On Mobile: Tap 'Browse files' -> Select 'Camera' or 'Camcorder'")
    
    # ONE UPLOADER FOR EVERYTHING
    uploaded_file = st.file_uploader(
        "Choose Image or Video", 
        type=['jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi'], 
        label_visibility="collapsed"
    )

    if uploaded_file:
        st.success("Processing...")
        
        # --- LOGIC A: IT IS AN IMAGE ---
        if uploaded_file.type.startswith('image'):
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            # Run Model (High Quality)
            results = model(img_array, conf=0.25)[0]
            
            # Show Result
            st.image(results.plot(), use_column_width=True, caption="Detected Waste")
            
        # --- LOGIC B: IT IS A VIDEO ---
        elif uploaded_file.type.startswith('video'):
            # Save video to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            
            # Process video frame by frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                results = model(frame, conf=0.25)[0]
                st_frame.image(results.plot(), channels="BGR")
            cap.release()