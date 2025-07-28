# PART B1

import numpy as np
from scipy.stats import norm  # For PDF calculations


# Custom GMM implementation for univariate data
def custom_GMM_uni(data, K_components, epsilon=1e-6, seed=None):
    # Set the random seed for reproducibility (if specified)
    if seed is not None:
        np.random.seed(seed)

    # Initialize model parameters: weights (omega), means (mu), and variances (Sigma)
    n_samples = len(data)
    omega = np.full(K_components, 1 / K_components)  # Equal initial weights
    mu = np.random.choice(
        data, K_components, replace=False
    )  # Randomly pick K means from data
    Sigma = np.full(
        K_components, np.var(data)
    )  # Initialize all variances to the variance of the data

    # Iterative EM Algorithm
    log_likelihood_prev = -np.inf  # Initial log-likelihood (a very low starting value)
    while True:
        # E-step: Compute responsibilities (probabilities of each sample belonging to each component)
        responsibilities = np.zeros((n_samples, K_components))
        for k in range(K_components):
            responsibilities[:, k] = omega[k] * norm.pdf(
                data, loc=mu[k], scale=np.sqrt(Sigma[k])
            )
        responsibilities /= np.sum(
            responsibilities, axis=1, keepdims=True
        )  # Normalize responsibilities

        # M-step: Update parameters based on the responsibilities
        N_k = np.sum(
            responsibilities, axis=0
        )  # Effective number of samples per component
        omega = N_k / n_samples  # Update weights (proportions of each component)
        mu = (
            np.sum(responsibilities * data[:, np.newaxis], axis=0) / N_k
        )  # Update means
        Sigma = (
            np.sum(responsibilities * (data[:, np.newaxis] - mu) ** 2, axis=0) / N_k
        )  # Update variances

        # Compute log-likelihood of the current model
        log_likelihood = np.sum(
            np.log(
                np.sum(
                    [
                        omega[k] * norm.pdf(data, loc=mu[k], scale=np.sqrt(Sigma[k]))
                        for k in range(K_components)
                    ],
                    axis=0,
                )
            )
        )

        # Check for convergence by comparing log-likelihood to previous value
        if np.abs(log_likelihood - log_likelihood_prev) < epsilon:
            break  # Stop if the change in log-likelihood is smaller than epsilon
        log_likelihood_prev = log_likelihood  # Update the previous log-likelihood

    # Sort the components by their means for consistent output
    sorted_indices = np.argsort(mu)  # Sort indices based on the means
    omega = omega[sorted_indices]  # Reorder weights
    mu = mu[sorted_indices]  # Reorder means
    Sigma = Sigma[sorted_indices]  # Reorder variances

    # Format and return the model parameters in a dictionary
    params_dict = {
        "omega": np.round(omega, 2),  # Fitted weights (rounded to two decimal places)
        "mu": np.round(mu, 2),  # Fitted means (rounded to two decimal places)
        "Sigma": np.round(Sigma, 2),  # Fitted variances (rounded to two decimal places)
    }
    return params_dict  # Return the dictionary containing the model parameters


# PART B2

import numpy as np
from scipy.stats import multivariate_normal


def custom_GMM_multi(data, K_components, epsilon=1e-6, seed=None):
    """
    Custom implementation of a Gaussian Mixture Model (GMM) for multivariate data.
    This function applies the Expectation-Maximization (EM) algorithm.
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    n_samples, n_features = data.shape  # Get number of samples and features

    # Initialize model parameters
    omega = (
        np.ones(K_components) / K_components
    )  # Initialize weights equally for all components
    mu = data[
        np.random.choice(n_samples, K_components, replace=False)
    ]  # Randomly pick K means from data
    Sigma = np.array(
        [
            np.cov(data, rowvar=False) + 1e-6 * np.eye(n_features)
            for _ in range(K_components)
        ]
    )  # Initialize covariance matrices with small regularization

    log_likelihood_prev = -np.inf  # Set initial log-likelihood value
    for iteration in range(100):  # Maximum of 100 iterations
        # E-step: Compute responsibilities (probabilities of each sample belonging to each component)
        responsibilities = np.zeros((n_samples, K_components))
        for k in range(K_components):
            responsibilities[:, k] = omega[k] * multivariate_normal.pdf(
                data,
                mean=mu[k],
                cov=Sigma[k],
                allow_singular=True,  # Compute Gaussian PDF for each component
            )
        responsibilities /= responsibilities.sum(
            axis=1, keepdims=True
        )  # Normalize responsibilities for each sample

        # M-step: Update model parameters based on responsibilities
        N_k = responsibilities.sum(
            axis=0
        )  # Sum of responsibilities for each component (effective number of samples per component)
        omega = N_k / n_samples  # Update weights (proportions of each component)
        mu = (
            np.dot(responsibilities.T, data) / N_k[:, np.newaxis]
        )  # Update means (weighted average of samples)

        # Update covariance matrices for each component with regularization
        Sigma = np.array(
            [
                np.dot(
                    (responsibilities[:, k][:, np.newaxis] * (data - mu[k])).T,
                    (data - mu[k]),
                )
                / N_k[k]
                + 1e-6
                * np.eye(
                    n_features
                )  # Add small regularization to covariance matrices to avoid singular matrices
                for k in range(K_components)
            ]
        )

        # Compute log-likelihood of the current model
        log_likelihood = np.sum(
            np.log(
                np.sum(
                    [
                        omega[k]
                        * multivariate_normal.pdf(
                            data, mean=mu[k], cov=Sigma[k], allow_singular=True
                        )
                        for k in range(K_components)
                    ],
                    axis=0,
                )
            )
        )

        # Check for convergence (if the log-likelihood has stopped improving)
        if np.abs(log_likelihood - log_likelihood_prev) < epsilon:
            break  # Stop if the improvement in log-likelihood is smaller than the threshold
        log_likelihood_prev = log_likelihood  # Update the previous log-likelihood

    # Return the model parameters, rounded to two decimal places for consistent output
    return {
        "omega": np.round(omega, 2),  # Weights of each component
        "mu": np.round(mu, 2),  # Means of each component
        "Sigma": np.round(Sigma, 2),  # Covariance matrices of each component
    }


# modified A

import numpy as np
import librosa
from tqdm import tqdm


# Function to perform speaker recognition using custom GMMs
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

    # Train a GMM for each speaker using custom GMM implementation
    for speaker, files in tqdm(speaker_files.items(), desc="Training Speakers"):
        speaker_features = [
            extract_features(file) for file in files
        ]  # Extract features from all files

        if speaker_features:  # Check if features are available for training
            speaker_features = np.vstack(
                speaker_features
            )  # Stack features into a single array

            # Train GMM using custom GMM implementation (without sklearn)
            params = custom_GMM_multi(
                speaker_features, K_components=8, epsilon=1e-6, seed=42
            )
            speaker_models[speaker] = params  # Store trained GMM parameters

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
        log_likelihoods = {}
        for speaker, params in speaker_models.items():
            omega, mu, Sigma = params["omega"], params["mu"], params["Sigma"]
            log_likelihoods[speaker] = 0  # Initialize log-likelihood for the speaker

            for k in range(len(omega)):
                # Manually calculate log-likelihood for each Gaussian component
                diff = test_features - mu[k]
                inv_cov = np.linalg.inv(Sigma[k])  # Inverse of the covariance matrix
                log_det_cov = np.linalg.det(
                    Sigma[k]
                )  # Determinant of the covariance matrix

                # Calculate the quadratic form: (x - mu)^T Σ^-1 (x - mu)
                quad_form = np.sum(np.dot(diff, inv_cov) * diff, axis=1)

                # Compute log-likelihood for this component and sum over all test features
                log_likelihoods[speaker] += omega[k] * (
                    -0.5
                    * (
                        np.log(2 * np.pi) * test_features.shape[1]
                        + np.log(log_det_cov)
                        + quad_form
                    )
                )

        # After calculating the log-likelihood for each test file, get the speaker with the highest log-likelihood
        predicted_speaker = max(
            log_likelihoods, key=lambda speaker: log_likelihoods[speaker].sum()
        )  # Ensure scalar sum for comparison

        # Store the prediction with the test file name as key
        predict_dict[test_file.split("/")[-1]] = predicted_speaker

    return predict_dict  # Return predictions as a dictionary
