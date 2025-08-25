#!/usr/bin/env python3
"""
Adversarial Training Module for Robust Speaker Recognition

Implements adversarial training techniques to improve robustness against:
- Audio noise and distortions
- Channel variations and recording conditions
- Spoofing attacks and voice conversion
- Domain shifts and acoustic mismatches

Based on research from:
- "Adversarial Training for Speaker Recognition" (Li et al., 2020)
- "Robust Speaker Verification via Adversarial Training" (Wang et al., 2019)
- "Anti-spoofing for Speaker Recognition" (Wu et al., 2021)
- "Domain Adversarial Training" (Ganin et al., 2016)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging
import random
from scipy import signal
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

class AudioAugmentationGenerator:
    """
    Generate adversarial audio augmentations for robust training.
    
    Includes various types of audio distortions and noise patterns
    that speakers might encounter in real-world scenarios.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def generate_adversarial_audio(self, 
                                 audio: np.ndarray, 
                                 augmentation_type: str = "random",
                                 intensity: float = 0.3) -> np.ndarray:
        """
        Generate adversarial audio with specified augmentation.
        
        Args:
            audio: Original audio waveform
            augmentation_type: Type of augmentation to apply
            intensity: Intensity of augmentation (0.0 to 1.0)
            
        Returns:
            Augmented audio waveform
        """
        if augmentation_type == "random":
            augmentation_type = random.choice([
                "gaussian_noise", "pink_noise", "reverb", "compression",
                "pitch_shift", "time_stretch", "bandpass_filter", 
                "clipping", "dropout", "channel_simulation"
            ])
        
        try:
            if augmentation_type == "gaussian_noise":
                return self._add_gaussian_noise(audio, intensity)
            elif augmentation_type == "pink_noise":
                return self._add_pink_noise(audio, intensity)
            elif augmentation_type == "reverb":
                return self._add_reverb(audio, intensity)
            elif augmentation_type == "compression":
                return self._apply_compression(audio, intensity)
            elif augmentation_type == "pitch_shift":
                return self._pitch_shift(audio, intensity)
            elif augmentation_type == "time_stretch":
                return self._time_stretch(audio, intensity)
            elif augmentation_type == "bandpass_filter":
                return self._bandpass_filter(audio, intensity)
            elif augmentation_type == "clipping":
                return self._apply_clipping(audio, intensity)
            elif augmentation_type == "dropout":
                return self._apply_dropout(audio, intensity)
            elif augmentation_type == "channel_simulation":
                return self._simulate_channel_effects(audio, intensity)
            else:
                logger.warning(f"Unknown augmentation type: {augmentation_type}")
                return audio
                
        except Exception as e:
            logger.warning(f"⚠️ Augmentation {augmentation_type} failed: {e}")
            return audio
    
    def _add_gaussian_noise(self, audio: np.ndarray, intensity: float) -> np.ndarray:
        """Add Gaussian noise to audio."""
        noise_power = intensity * 0.1  # Scale noise power
        noise = np.random.normal(0, noise_power, audio.shape)
        return audio + noise
    
    def _add_pink_noise(self, audio: np.ndarray, intensity: float) -> np.ndarray:
        """Add pink noise (1/f noise) to audio."""
        # Generate white noise
        white_noise = np.random.normal(0, 1, len(audio))
        
        # Apply 1/f filter to create pink noise
        freqs = np.fft.fftfreq(len(white_noise), 1/self.sample_rate)
        freqs[0] = 1  # Avoid division by zero
        
        # Create 1/f filter
        filter_response = 1 / np.sqrt(np.abs(freqs))
        filter_response[0] = 0  # DC component
        
        # Apply filter
        white_fft = np.fft.fft(white_noise)
        pink_fft = white_fft * filter_response
        pink_noise = np.real(np.fft.ifft(pink_fft))
        
        # Scale and add to audio
        pink_noise = pink_noise * intensity * 0.05
        return audio + pink_noise
    
    def _add_reverb(self, audio: np.ndarray, intensity: float) -> np.ndarray:
        """Add simple reverb effect."""
        # Create impulse response for reverb
        reverb_length = int(0.5 * self.sample_rate)  # 0.5 second reverb
        impulse = np.zeros(reverb_length)
        
        # Add multiple delayed copies with exponential decay
        delays = [0.05, 0.1, 0.2, 0.3, 0.4]  # Delay times in seconds
        for delay in delays:
            delay_samples = int(delay * self.sample_rate)
            if delay_samples < len(impulse):
                decay = np.exp(-delay * 3)  # Exponential decay
                impulse[delay_samples] = intensity * decay * 0.3
        
        # Convolve with impulse response
        reverb_audio = signal.convolve(audio, impulse, mode='same')
        
        # Mix with original
        return 0.7 * audio + 0.3 * reverb_audio
    
    def _apply_compression(self, audio: np.ndarray, intensity: float) -> np.ndarray:
        """Apply dynamic range compression."""
        # Simple compression using tanh
        threshold = 0.5 * (1 - intensity)  # Lower threshold = more compression
        compressed = np.where(
            np.abs(audio) > threshold,
            np.sign(audio) * (threshold + (np.abs(audio) - threshold) * 0.3),
            audio
        )
        return compressed
    
    def _pitch_shift(self, audio: np.ndarray, intensity: float) -> np.ndarray:
        """Apply pitch shifting (simplified version)."""
        # Simple pitch shift using resampling
        shift_factor = 1 + (intensity - 0.5) * 0.2  # ±10% pitch shift
        
        # Resample to change pitch
        new_length = int(len(audio) / shift_factor)
        indices = np.linspace(0, len(audio) - 1, new_length)
        shifted = np.interp(indices, np.arange(len(audio)), audio)
        
        # Pad or trim to original length
        if len(shifted) < len(audio):
            shifted = np.pad(shifted, (0, len(audio) - len(shifted)), 'constant')
        else:
            shifted = shifted[:len(audio)]
        
        return shifted
    
    def _time_stretch(self, audio: np.ndarray, intensity: float) -> np.ndarray:
        """Apply time stretching without pitch change."""
        # Simple time stretching using interpolation
        stretch_factor = 1 + (intensity - 0.5) * 0.3  # ±15% time stretch
        
        new_length = int(len(audio) * stretch_factor)
        indices = np.linspace(0, len(audio) - 1, new_length)
        stretched = np.interp(indices, np.arange(len(audio)), audio)
        
        # Trim or pad to original length
        if len(stretched) > len(audio):
            stretched = stretched[:len(audio)]
        else:
            stretched = np.pad(stretched, (0, len(audio) - len(stretched)), 'constant')
        
        return stretched
    
    def _bandpass_filter(self, audio: np.ndarray, intensity: float) -> np.ndarray:
        """Apply bandpass filtering to simulate channel effects."""
        # Define filter parameters based on intensity
        low_freq = 300 + intensity * 200   # 300-500 Hz low cutoff
        high_freq = 3400 - intensity * 400  # 3000-3400 Hz high cutoff
        
        # Design Butterworth bandpass filter
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        if high >= 1.0:
            high = 0.99
        if low >= high:
            low = high - 0.1
        
        try:
            b, a = signal.butter(4, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, audio)
            return filtered
        except Exception:
            return audio
    
    def _apply_clipping(self, audio: np.ndarray, intensity: float) -> np.ndarray:
        """Apply audio clipping distortion."""
        clip_level = 1.0 - intensity * 0.5  # Reduce clipping level
        clipped = np.clip(audio, -clip_level, clip_level)
        return clipped
    
    def _apply_dropout(self, audio: np.ndarray, intensity: float) -> np.ndarray:
        """Apply random dropout to audio samples."""
        dropout_rate = intensity * 0.1  # Up to 10% dropout
        mask = np.random.random(len(audio)) > dropout_rate
        return audio * mask
    
    def _simulate_channel_effects(self, audio: np.ndarray, intensity: float) -> np.ndarray:
        """Simulate various channel effects."""
        # Combine multiple channel effects
        
        # 1. Frequency response variation
        freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)
        audio_fft = np.fft.fft(audio)
        
        # Create random frequency response
        response = 1 + intensity * 0.3 * np.random.normal(0, 1, len(freqs))
        response = np.maximum(response, 0.1)  # Ensure positive
        
        modified_fft = audio_fft * response
        modified_audio = np.real(np.fft.ifft(modified_fft))
        
        # 2. Add slight echo
        echo_delay = int(0.02 * self.sample_rate)  # 20ms echo
        echo = np.zeros_like(modified_audio)
        if echo_delay < len(echo):
            echo[echo_delay:] = modified_audio[:-echo_delay] * intensity * 0.2
        
        return modified_audio + echo

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer for adversarial training.
    
    Forward pass: identity function
    Backward pass: reverses gradients and scales by alpha
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        ctx.alpha = alpha
        return x
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.alpha * grad_output, None

class GradientReversalLayer(nn.Module):
    """Gradient reversal layer module."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        return GradientReversalFunction.apply(x, alpha)

class AdversarialEmbeddingNetwork(nn.Module):
    """
    Neural network for adversarial training of speaker embeddings.
    
    Uses gradient reversal layer to learn domain-invariant features
    while maintaining speaker discriminability.
    """
    
    def __init__(self, 
                 embedding_dim: int = 192,
                 hidden_dim: int = 256,
                 num_domains: int = 5):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_domains = num_domains
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2)
        )
        
        # Speaker classifier (main task)
        self.speaker_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, embedding_dim)  # Output embedding
        )
        
        # Domain classifier (adversarial task)
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_domains)
        )
        
        # Gradient reversal layer
        self.gradient_reversal = GradientReversalLayer()
    
    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with adversarial training.
        
        Args:
            x: Input embeddings
            alpha: Gradient reversal strength
            
        Returns:
            Speaker embeddings and domain predictions
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Speaker classification (main task)
        speaker_output = self.speaker_classifier(features)
        
        # Domain classification (adversarial task)
        reversed_features = self.gradient_reversal(features, alpha)
        domain_output = self.domain_classifier(reversed_features)
        
        return speaker_output, domain_output

class SpoofingDetectionNetwork(nn.Module):
    """
    Network for detecting spoofed/synthetic speech.
    
    Helps improve robustness against voice conversion and synthesis attacks.
    """
    
    def __init__(self, embedding_dim: int = 192):
        super().__init__()
        
        self.detector = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)  # Real vs. Spoofed
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Detect if embedding is from real or spoofed speech."""
        return self.detector(x)

class AdversarialTrainingSystem:
    """
    Complete adversarial training system for robust speaker recognition.
    
    Integrates audio augmentation, adversarial networks, and spoofing detection
    for comprehensive robustness against various attacks and conditions.
    """
    
    def __init__(self, 
                 embedding_dim: int = 192,
                 sample_rate: int = 16000,
                 enable_audio_augmentation: bool = True,
                 enable_domain_adaptation: bool = True,
                 enable_spoofing_detection: bool = True):
        """
        Initialize adversarial training system.
        
        Args:
            embedding_dim: Dimension of speaker embeddings
            sample_rate: Audio sample rate
            enable_audio_augmentation: Enable audio augmentation
            enable_domain_adaptation: Enable domain adversarial training
            enable_spoofing_detection: Enable spoofing detection
        """
        self.embedding_dim = embedding_dim
        self.sample_rate = sample_rate
        self.enable_audio_augmentation = enable_audio_augmentation
        self.enable_domain_adaptation = enable_domain_adaptation
        self.enable_spoofing_detection = enable_spoofing_detection
        
        # Initialize components
        self.audio_augmenter = AudioAugmentationGenerator(sample_rate) if enable_audio_augmentation else None
        
        self.adversarial_network = AdversarialEmbeddingNetwork(embedding_dim) if enable_domain_adaptation else None
        
        self.spoofing_detector = SpoofingDetectionNetwork(embedding_dim) if enable_spoofing_detection else None
        
        # Training statistics
        self.training_stats = {
            "augmentations_applied": 0,
            "adversarial_loss": 0.0,
            "spoofing_accuracy": 0.0,
            "domain_accuracy": 0.0
        }
        
        logger.info(f"🛡️ Initialized adversarial training system "
                   f"(augmentation={enable_audio_augmentation}, "
                   f"domain_adapt={enable_domain_adaptation}, "
                   f"spoofing={enable_spoofing_detection})")
    
    def apply_adversarial_augmentation(self, 
                                     audio: np.ndarray,
                                     augmentation_strength: float = 0.3) -> np.ndarray:
        """
        Apply adversarial audio augmentation.
        
        Args:
            audio: Original audio waveform
            augmentation_strength: Strength of augmentation (0.0 to 1.0)
            
        Returns:
            Augmented audio waveform
        """
        if not self.enable_audio_augmentation or self.audio_augmenter is None:
            return audio
        
        try:
            # Apply random augmentation
            augmented = self.audio_augmenter.generate_adversarial_audio(
                audio, "random", augmentation_strength
            )
            
            self.training_stats["augmentations_applied"] += 1
            return augmented
            
        except Exception as e:
            logger.warning(f"⚠️ Adversarial augmentation failed: {e}")
            return audio
    
    def robust_embedding_extraction(self, 
                                  base_embedding: np.ndarray,
                                  domain_id: Optional[int] = None) -> np.ndarray:
        """
        Extract robust embedding using adversarial training.
        
        Args:
            base_embedding: Base speaker embedding
            domain_id: Domain identifier for adversarial training
            
        Returns:
            Robust speaker embedding
        """
        if not self.enable_domain_adaptation or self.adversarial_network is None:
            return base_embedding
        
        try:
            embedding_tensor = torch.from_numpy(base_embedding).float().unsqueeze(0)
            
            with torch.no_grad():
                robust_embedding, domain_pred = self.adversarial_network(embedding_tensor)
                robust_embedding = robust_embedding.squeeze(0).numpy()
            
            return robust_embedding
            
        except Exception as e:
            logger.warning(f"⚠️ Robust embedding extraction failed: {e}")
            return base_embedding
    
    def detect_spoofing(self, embedding: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if embedding is from spoofed speech.
        
        Args:
            embedding: Speaker embedding to analyze
            
        Returns:
            (is_spoofed, confidence_score)
        """
        if not self.enable_spoofing_detection or self.spoofing_detector is None:
            return False, 0.0
        
        try:
            embedding_tensor = torch.from_numpy(embedding).float().unsqueeze(0)
            
            with torch.no_grad():
                spoofing_logits = self.spoofing_detector(embedding_tensor)
                spoofing_probs = F.softmax(spoofing_logits, dim=1)
                
                # Class 1 is spoofed, class 0 is real
                is_spoofed = spoofing_probs[0, 1] > 0.5
                confidence = spoofing_probs[0, 1].item()
            
            return is_spoofed.item(), confidence
            
        except Exception as e:
            logger.warning(f"⚠️ Spoofing detection failed: {e}")
            return False, 0.0
    
    def get_training_status(self) -> Dict:
        """Get comprehensive training status and statistics."""
        return {
            "configuration": {
                "embedding_dim": self.embedding_dim,
                "sample_rate": self.sample_rate,
                "audio_augmentation": self.enable_audio_augmentation,
                "domain_adaptation": self.enable_domain_adaptation,
                "spoofing_detection": self.enable_spoofing_detection
            },
            "statistics": self.training_stats.copy()
        }

# Factory function
def create_adversarial_system(**kwargs) -> AdversarialTrainingSystem:
    """Create an adversarial training system with custom configuration."""
    return AdversarialTrainingSystem(**kwargs)