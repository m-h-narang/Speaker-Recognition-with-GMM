# PART A

import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


# Function to perform speaker recognition using GMMs
def speaker_rec_GMM(audio_dir, test_dir):

    # Extract features (MFCC) from an audio file
    def extract_features(file_path):
        audio, sr = librosa.load(file_path, sr=None)  # Load the audio file
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=13
        )  # Extract 13 MFCC features
        return mfcc.T  # Transpose to match GMM input format

    # Group audio files by speaker label based on the directory structure
    def group_files_by_speaker(audio_dir):
        all_files = librosa.util.find_files(audio_dir, ext="wav")  # Find .wav files
        speaker_files = {}  # Store files grouped by speaker

        for file_path in all_files:
            speaker = file_path.split("/")[
                -2
            ]  # Extract speaker label from parent directory
            if speaker not in speaker_files:
                speaker_files[speaker] = []
            speaker_files[speaker].append(file_path)  # Group files for each speaker

        return speaker_files

    speaker_models = {}  # Store GMMs for each speaker
    speaker_files = group_files_by_speaker(audio_dir)  # Group training files by speaker

    # Train a GMM for each speaker
    for speaker, files in tqdm(speaker_files.items(), desc="Training Speakers"):
        speaker_features = [
            extract_features(file) for file in files
        ]  # Extract features from all files

        if speaker_features:  # Check if features are available for training
            speaker_features = np.vstack(
                speaker_features
            )  # Stack features into a single array

            # Initialize GMM with 8 components and diagonal covariance
            gmm = GaussianMixture(
                n_components=8, covariance_type="diag", max_iter=200, random_state=42
            )
            gmm.fit(speaker_features)  # Train the GMM
            speaker_models[speaker] = gmm  # Store trained GMM

    predict_dict = {}  # Store predictions for test files
    test_files = librosa.util.find_files(
        test_dir, ext="wav"
    )  # Find .wav files in the test directory

    # Predict speaker for each test file
    for test_file in tqdm(test_files, desc="Testing"):
        test_features = extract_features(
            test_file
        )  # Extract features for the test file

        # Calculate log-likelihoods for each speaker's GMM
        log_likelihoods = {
            speaker: model.score(test_features)
            for speaker, model in speaker_models.items()
        }

        # Identify the speaker with the highest log-likelihood
        predicted_speaker = max(log_likelihoods, key=log_likelihoods.get)

        # Store the prediction with the test file name as key
        predict_dict[test_file.split("/")[-1]] = predicted_speaker

    return predict_dict  # Return predictions as a dictionary
