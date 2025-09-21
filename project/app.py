import streamlit as st
import cv2
import numpy as np
import os
import folium
from streamlit_folium import st_folium
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Compatibility for scikit-image versions ---
try:
    from skimage.feature import graycomatrix as greycomatrix, graycoprops as greycoprops
except ImportError:
    from skimage.feature import greycomatrix, greycoprops

# ------------------------
# Feature Extraction
# ------------------------
def fft_slope(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)

    center = np.array(magnitude_spectrum.shape) // 2
    y, x = np.indices(magnitude_spectrum.shape)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(np.int32)

    tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
    nr = np.bincount(r.ravel())
    radial_prof = tbin / (nr + 1e-8)

    r_nonzero = np.arange(1, len(radial_prof))
    slope, _ = np.polyfit(np.log(r_nonzero), np.log(radial_prof[1:]), 1)
    return slope

def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (256, 256))

    edges = cv2.Laplacian(img, cv2.CV_64F)
    edge_var = edges.var()

    glcm = greycomatrix(img, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]

    slope = fft_slope(img)
    return [edge_var, contrast, correlation, energy, slope]

# ------------------------
# Load Dataset from GitHub repo folder
# ------------------------
def load_dataset(base_dir="dataset"):
    X, y = [], []
    labels = {"fine": 0, "medium": 1, "coarse": 2}
    for cls, label in labels.items():
        folder = os.path.join(base_dir, cls)
        if not os.path.exists(folder):
            st.warning(f"Folder not found: {folder}")
            continue
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(files) == 0:
            st.warning(f"No images found in: {folder}")
        for fname in files:
            path = os.path.join(folder, fname)
            features = extract_features(path)
            if features is not None:
                X.append(features)
                y.append(label)
    st.write(f"Total images loaded: {len(X)}")
    return np.array(X), np.array(y)

# ------------------------
# Streamlit App
# ------------------------
st.title("Innovexa's Beach Sand Classification App")

uploaded_file = st.file_uploader("Upload a beach sand image", type=["jpg", "png", "jpeg"])
lat = st.text_input("Enter Latitude:")
lon = st.text_input("Enter Longitude:")

# Initialize session state
if "markers" not in st.session_state:
    st.session_state.markers = []
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ------------------------
# Load dataset and train model
# ------------------------
X, y = load_dataset("dataset")
if len(X) == 0:
    st.error("Dataset is empty! Add images in dataset/fine, dataset/medium, dataset/coarse.")
    st.stop()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
clf = SVC(kernel='rbf', C=10, gamma=0.1)
clf.fit(X_train, y_train)
classes = ["Fine", "Medium", "Coarse"]

# ------------------------
# Prediction
# ------------------------
if st.button("Predict Sand Classification"):
    if uploaded_file is None or not lat or not lon:
        st.warning("Please upload an image and enter latitude & longitude first.")
    else:
        temp_path = "temp_image.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        features = extract_features(temp_path)
        if features:
            features_scaled = scaler.transform([features])
            pred = clf.predict(features_scaled)[0]
            result = classes[pred]

            # Inference text
            if result == "Fine":
                inference = "Approx. Size: 0.125 – 0.25 mm\nBest for tourism/recreation.\nNot suitable for heavy ports."
            elif result == "Medium":
                inference = "Approx. Size: 0.25 – 0.5 mm\nSuitable for fishing harbors, small breakwaters.\nNot suitable for skyscrapers or nuclear plants."
            else:
                inference = "Approx. Size: 0.5 – 1 mm\nBest for ports/harbors/lighthouses.\nNot suitable for farming or tourist resorts."

            # Save last prediction
            st.session_state.last_prediction = (result, inference, temp_path)

            # Add map marker
            try:
                lat_f, lon_f = float(lat), float(lon)
                st.session_state.markers.append((lat_f, lon_f, result))
            except:
                st.error("Invalid latitude/longitude input.")
        else:
            st.error("Could not extract features from image.")

# ------------------------
# Display last prediction
# ------------------------
if st.session_state.last_prediction:
    result, inference, img_path = st.session_state.last_prediction
    st.success(f"Sand Grain Classification: {result}")
    for line in inference.split("\n"):
        st.write(line)
    st.image(img_path, caption=f"Uploaded Image - {result} Sand", use_container_width=True)

# ------------------------
# Display map with markers and legend
# ------------------------
if st.session_state.markers:
    last_marker = st.session_state.markers[-1]
    m = folium.Map(location=[last_marker[0], last_marker[1]], zoom_start=8)
    for mk in st.session_state.markers:
        folium.Marker(
            [mk[0], mk[1]],
            popup=f"{mk[2]} Sand",
            tooltip=f"{mk[2]} Sand",
            icon=folium.Icon(color="blue" if mk[2]=="Fine" else "orange" if mk[2]=="Medium" else "green")
        ).add_to(m)
    st_folium(m, width=700, height=500)

    st.markdown("""
    <div style='padding:10px; border:1px solid grey; width:200px;'>
    <b>Sand Type Legend</b><br>
    <span style='color:blue;'>●</span> Fine<br>
    <span style='color:orange;'>●</span> Medium<br>
    <span style='color:green;'>●</span> Coarse
    </div>
    """, unsafe_allow_html=True)
