import os
import librosa
import numpy as np
import noisereduce as nr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from collections import Counter


def reduce_noise(audio, sr):
    """Reduce noise from audio signal."""
    return nr.reduce_noise(y=audio, sr=sr)


def extract_features(audio, sr):
    """Extract audio features including pitch, bandwidth, MFCCs, etc."""
    try:
        # Pitch (using librosa.pyin)
        pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sr)
        pitch = np.max(pitches) if np.any(pitches) else 0.0

        # Bandwidth (using spectral bandwidth)
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr).mean()

        # MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)

        # Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr).mean()

        # Zero-Crossing Rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio).mean()

        features = np.hstack([
            pitch,
            bandwidth,
            mfccs_mean,
            spectral_centroid,
            zero_crossing_rate
        ])
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.zeros(17)


def load_data(directory):
    """Load audio data and extract features."""
    features = []
    labels = []
    for label in os.listdir(directory):
        label_dir = os.path.join(directory, label)
        if not os.path.isdir(label_dir):
            continue
        for file in os.listdir(label_dir):
            if file.endswith(".wav"):
                file_path = os.path.join(label_dir, file)
                audio, sr = librosa.load(file_path, sr=None)

                # Noise reduction
                audio_cleaned = reduce_noise(audio, sr)

                # Feature extraction
                feature_vector = extract_features(audio_cleaned, sr)
                features.append(feature_vector)
                labels.append(label)
    return np.array(features), np.array(labels)


def predict_animal(file_path, model, scaler):
    """Predict animal class for a given audio file."""
    audio, sr = librosa.load(file_path, sr=None)
    audio_cleaned = reduce_noise(audio, sr)
    feature_vector = extract_features(audio_cleaned, sr)
    feature_vector_scaled = scaler.transform([feature_vector])
    prediction = model.predict(feature_vector_scaled)
    return prediction[0]


def main():
    """Main function to train and evaluate the model."""
    data_dir = r"C:\Users\ediso\Downloads\dataset"
    features, labels = load_data(data_dir)

    print("Jumlah sampel per kelas:", Counter(labels))
    print("Contoh fitur yang diekstraksi:", features[:5])
    print("Apakah ada nilai NaN dalam fitur?", np.isnan(features).any())
    print("Apakah ada nilai infinite dalam fitur?", np.isinf(features).any())

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, labels, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Model evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Akurasi: {accuracy * 100:.2f}%")
    print("\nLaporan Klasifikasi:")
    print(classification_report(y_test, y_pred))

    test_file = r"C:\Users\ediso\Downloads\test_dog.wav"
    predicted_animal = predict_animal(test_file, model, scaler)
    print(f"Prediksi hewan untuk file {test_file}: {predicted_animal}")


if __name__ == "__main__":
    main()