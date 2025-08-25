#!/usr/bin/env python3
"""
Multimodal Fusion Module for Enhanced Speaker Recognition

Combines multiple modalities for improved speaker identification:
- Audio embeddings (ECAPA-TDNN, pyannote)
- Prosodic features (pitch, rhythm, stress patterns)
- Linguistic features (speaking style, vocabulary patterns)
- Temporal dynamics (speaking rate, pause patterns)

Based on research from:
- "Multimodal Speaker Recognition" (Nagrani et al., 2020)
- "Prosodic Features for Speaker Recognition" (Kockmann et al., 2010)
- "Linguistic Features in Speaker Recognition" (Doddington, 2001)
- "Cross-modal Speaker Verification" (Chung et al., 2020)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging
import librosa
from scipy import signal
from scipy.stats import skew, kurtosis
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

class ProsodyExtractor:
    """
    Extract prosodic features from audio for speaker characterization.
    
    Features include:
    - Fundamental frequency (F0) statistics and dynamics
    - Energy/intensity patterns
    - Speaking rate and rhythm
    - Pause patterns and timing
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def extract_prosodic_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract comprehensive prosodic features from audio.
        
        Args:
            audio: Audio waveform
            
        Returns:
            Dictionary of prosodic features
        """
        features = {}
        
        try:
            # F0 (pitch) features
            f0_features = self._extract_f0_features(audio)
            features.update(f0_features)
            
            # Energy features
            energy_features = self._extract_energy_features(audio)
            features.update(energy_features)
            
            # Rhythm and timing features
            rhythm_features = self._extract_rhythm_features(audio)
            features.update(rhythm_features)
            
            # Spectral features related to prosody
            spectral_features = self._extract_spectral_prosody_features(audio)
            features.update(spectral_features)
            
        except Exception as e:
            logger.warning(f"⚠️ Prosody extraction failed: {e}")
            # Return zero features as fallback
            features = {f"prosody_{i}": 0.0 for i in range(20)}
        
        return features
    
    def _extract_f0_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract fundamental frequency features."""
        try:
            # Extract F0 using librosa
            f0 = librosa.yin(audio, 
                           fmin=librosa.note_to_hz('C2'), 
                           fmax=librosa.note_to_hz('C7'),
                           sr=self.sample_rate)
            
            # Remove unvoiced frames (f0 = 0)
            voiced_f0 = f0[f0 > 0]
            
            if len(voiced_f0) == 0:
                return {f"f0_{feature}": 0.0 for feature in 
                       ['mean', 'std', 'min', 'max', 'range', 'skew', 'kurt']}
            
            features = {
                'f0_mean': np.mean(voiced_f0),
                'f0_std': np.std(voiced_f0),
                'f0_min': np.min(voiced_f0),
                'f0_max': np.max(voiced_f0),
                'f0_range': np.max(voiced_f0) - np.min(voiced_f0),
                'f0_skew': skew(voiced_f0),
                'f0_kurt': kurtosis(voiced_f0)
            }
            
            # F0 dynamics (rate of change)
            f0_diff = np.diff(voiced_f0)
            features['f0_jitter'] = np.std(f0_diff) if len(f0_diff) > 0 else 0.0
            
            return features
            
        except Exception as e:
            logger.debug(f"F0 extraction failed: {e}")
            return {f"f0_{feature}": 0.0 for feature in 
                   ['mean', 'std', 'min', 'max', 'range', 'skew', 'kurt', 'jitter']}
    
    def _extract_energy_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract energy and intensity features."""
        try:
            # RMS energy
            frame_length = int(0.025 * self.sample_rate)  # 25ms frames
            hop_length = int(0.010 * self.sample_rate)    # 10ms hop
            
            rms = librosa.feature.rms(y=audio, 
                                    frame_length=frame_length, 
                                    hop_length=hop_length)[0]
            
            # Convert to dB
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            
            features = {
                'energy_mean': np.mean(rms_db),
                'energy_std': np.std(rms_db),
                'energy_range': np.max(rms_db) - np.min(rms_db),
                'energy_skew': skew(rms_db),
            }
            
            # Energy dynamics
            energy_diff = np.diff(rms_db)
            features['energy_shimmer'] = np.std(energy_diff) if len(energy_diff) > 0 else 0.0
            
            return features
            
        except Exception as e:
            logger.debug(f"Energy extraction failed: {e}")
            return {f"energy_{feature}": 0.0 for feature in 
                   ['mean', 'std', 'range', 'skew', 'shimmer']}
    
    def _extract_rhythm_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract rhythm and timing features."""
        try:
            # Voice activity detection for rhythm analysis
            frame_length = int(0.025 * self.sample_rate)
            hop_length = int(0.010 * self.sample_rate)
            
            # Simple energy-based VAD
            rms = librosa.feature.rms(y=audio, 
                                    frame_length=frame_length, 
                                    hop_length=hop_length)[0]
            
            # Threshold for voice activity
            threshold = np.mean(rms) * 0.3
            voice_activity = rms > threshold
            
            # Speaking rate (voiced frames per second)
            speaking_rate = np.sum(voice_activity) / (len(voice_activity) * hop_length / self.sample_rate)
            
            # Pause analysis
            pause_frames = ~voice_activity
            pause_segments = self._get_segments(pause_frames)
            
            if pause_segments:
                pause_durations = [(end - start) * hop_length / self.sample_rate 
                                 for start, end in pause_segments]
                avg_pause_duration = np.mean(pause_durations)
                pause_frequency = len(pause_segments) / (len(audio) / self.sample_rate)
            else:
                avg_pause_duration = 0.0
                pause_frequency = 0.0
            
            features = {
                'speaking_rate': speaking_rate,
                'avg_pause_duration': avg_pause_duration,
                'pause_frequency': pause_frequency,
                'voice_activity_ratio': np.mean(voice_activity)
            }
            
            return features
            
        except Exception as e:
            logger.debug(f"Rhythm extraction failed: {e}")
            return {f"rhythm_{feature}": 0.0 for feature in 
                   ['speaking_rate', 'avg_pause_duration', 'pause_frequency', 'voice_activity_ratio']}
    
    def _extract_spectral_prosody_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract spectral features related to prosody."""
        try:
            # Spectral centroid (brightness)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            
            # Zero crossing rate (related to voicing)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            
            features = {
                'spectral_centroid_mean': np.mean(spectral_centroids),
                'spectral_centroid_std': np.std(spectral_centroids),
                'spectral_rolloff_mean': np.mean(spectral_rolloff),
                'zcr_mean': np.mean(zcr),
                'zcr_std': np.std(zcr)
            }
            
            return features
            
        except Exception as e:
            logger.debug(f"Spectral prosody extraction failed: {e}")
            return {f"spectral_{feature}": 0.0 for feature in 
                   ['centroid_mean', 'centroid_std', 'rolloff_mean', 'zcr_mean', 'zcr_std']}
    
    def _get_segments(self, binary_array: np.ndarray) -> List[Tuple[int, int]]:
        """Get continuous segments from binary array."""
        segments = []
        start = None
        
        for i, val in enumerate(binary_array):
            if val and start is None:
                start = i
            elif not val and start is not None:
                segments.append((start, i))
                start = None
        
        if start is not None:
            segments.append((start, len(binary_array)))
        
        return segments

class LinguisticFeatureExtractor:
    """
    Extract linguistic features from transcribed text for speaker characterization.
    
    Features include:
    - Vocabulary richness and complexity
    - Sentence structure patterns
    - Speaking style indicators
    - Disfluency patterns
    """
    
    def __init__(self):
        # Common function words for style analysis
        self.function_words = {
            'articles': ['a', 'an', 'the'],
            'prepositions': ['in', 'on', 'at', 'by', 'for', 'with', 'to', 'from', 'of', 'about'],
            'pronouns': ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'],
            'conjunctions': ['and', 'or', 'but', 'so', 'because', 'although', 'while', 'if'],
            'auxiliary_verbs': ['is', 'are', 'was', 'were', 'have', 'has', 'had', 'will', 'would', 'can', 'could']
        }
        
        # Disfluency markers
        self.disfluency_markers = ['um', 'uh', 'er', 'ah', 'like', 'you know', 'i mean']
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """
        Extract linguistic features from transcribed text.
        
        Args:
            text: Transcribed text
            
        Returns:
            Dictionary of linguistic features
        """
        if not text or not text.strip():
            return {f"linguistic_{i}": 0.0 for i in range(15)}
        
        features = {}
        
        try:
            # Basic text statistics
            basic_features = self._extract_basic_features(text)
            features.update(basic_features)
            
            # Vocabulary features
            vocab_features = self._extract_vocabulary_features(text)
            features.update(vocab_features)
            
            # Style features
            style_features = self._extract_style_features(text)
            features.update(style_features)
            
            # Disfluency features
            disfluency_features = self._extract_disfluency_features(text)
            features.update(disfluency_features)
            
        except Exception as e:
            logger.warning(f"⚠️ Linguistic feature extraction failed: {e}")
            features = {f"linguistic_{i}": 0.0 for i in range(15)}
        
        return features
    
    def _extract_basic_features(self, text: str) -> Dict[str, float]:
        """Extract basic text statistics."""
        words = text.lower().split()
        sentences = text.split('.')
        
        if not words:
            return {'avg_word_length': 0.0, 'avg_sentence_length': 0.0, 'total_words': 0.0}
        
        features = {
            'avg_word_length': np.mean([len(word) for word in words]),
            'avg_sentence_length': len(words) / max(len(sentences), 1),
            'total_words': len(words)
        }
        
        return features
    
    def _extract_vocabulary_features(self, text: str) -> Dict[str, float]:
        """Extract vocabulary richness features."""
        words = text.lower().split()
        
        if not words:
            return {'vocabulary_richness': 0.0, 'hapax_ratio': 0.0}
        
        # Type-token ratio (vocabulary richness)
        unique_words = set(words)
        vocabulary_richness = len(unique_words) / len(words)
        
        # Hapax legomena ratio (words appearing only once)
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        hapax_words = [word for word, count in word_counts.items() if count == 1]
        hapax_ratio = len(hapax_words) / len(unique_words) if unique_words else 0.0
        
        features = {
            'vocabulary_richness': vocabulary_richness,
            'hapax_ratio': hapax_ratio
        }
        
        return features
    
    def _extract_style_features(self, text: str) -> Dict[str, float]:
        """Extract speaking style features."""
        words = text.lower().split()
        
        if not words:
            return {f"{category}_ratio": 0.0 for category in self.function_words.keys()}
        
        features = {}
        
        # Function word ratios
        for category, word_list in self.function_words.items():
            count = sum(1 for word in words if word in word_list)
            features[f"{category}_ratio"] = count / len(words)
        
        return features
    
    def _extract_disfluency_features(self, text: str) -> Dict[str, float]:
        """Extract disfluency patterns."""
        text_lower = text.lower()
        words = text_lower.split()
        
        if not words:
            return {'disfluency_rate': 0.0, 'repetition_rate': 0.0}
        
        # Count disfluency markers
        disfluency_count = sum(1 for marker in self.disfluency_markers 
                             if marker in text_lower)
        
        # Count word repetitions (simple heuristic)
        repetition_count = 0
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                repetition_count += 1
        
        features = {
            'disfluency_rate': disfluency_count / len(words),
            'repetition_rate': repetition_count / len(words)
        }
        
        return features

class MultimodalFusionNetwork(nn.Module):
    """
    Neural network for fusing multimodal features for speaker recognition.
    
    Combines audio embeddings, prosodic features, and linguistic features
    using attention-based fusion and cross-modal learning.
    """
    
    def __init__(self, 
                 audio_dim: int = 192,
                 prosody_dim: int = 25,
                 linguistic_dim: int = 15,
                 fusion_dim: int = 256):
        super().__init__()
        
        self.audio_dim = audio_dim
        self.prosody_dim = prosody_dim
        self.linguistic_dim = linguistic_dim
        self.fusion_dim = fusion_dim
        
        # Modality-specific encoders
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, fusion_dim),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_dim),
            nn.Dropout(0.2)
        )
        
        self.prosody_encoder = nn.Sequential(
            nn.Linear(prosody_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.Linear(fusion_dim // 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.linguistic_encoder = nn.Sequential(
            nn.Linear(linguistic_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.Linear(fusion_dim // 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(fusion_dim, num_heads=8, batch_first=True)
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_dim * 2),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_dim),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, fusion_dim)
        )
        
        # Output projection
        self.output_projection = nn.Linear(fusion_dim, audio_dim)  # Project back to embedding space
    
    def forward(self, 
                audio_features: torch.Tensor,
                prosody_features: torch.Tensor,
                linguistic_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse multimodal features.
        
        Args:
            audio_features: Audio embeddings [batch_size, audio_dim]
            prosody_features: Prosodic features [batch_size, prosody_dim]
            linguistic_features: Linguistic features [batch_size, linguistic_dim]
            
        Returns:
            Fused speaker representation [batch_size, audio_dim]
        """
        # Encode each modality
        audio_encoded = self.audio_encoder(audio_features)
        prosody_encoded = self.prosody_encoder(prosody_features)
        linguistic_encoded = self.linguistic_encoder(linguistic_features)
        
        # Stack for attention
        modalities = torch.stack([audio_encoded, prosody_encoded, linguistic_encoded], dim=1)
        
        # Apply cross-modal attention
        attended, _ = self.cross_attention(modalities, modalities, modalities)
        
        # Flatten for fusion
        attended_flat = attended.view(attended.size(0), -1)
        
        # Fuse modalities
        fused = self.fusion_layers(attended_flat)
        
        # Project to output space
        output = self.output_projection(fused)
        
        return output

class MultimodalSpeakerSystem:
    """
    Complete multimodal speaker recognition system.
    
    Integrates audio embeddings with prosodic and linguistic features
    for enhanced speaker identification and verification.
    """
    
    def __init__(self, 
                 audio_dim: int = 192,
                 enable_prosody: bool = True,
                 enable_linguistic: bool = True,
                 sample_rate: int = 16000):
        """
        Initialize multimodal speaker system.
        
        Args:
            audio_dim: Dimension of audio embeddings
            enable_prosody: Enable prosodic feature extraction
            enable_linguistic: Enable linguistic feature extraction
            sample_rate: Audio sample rate
        """
        self.audio_dim = audio_dim
        self.enable_prosody = enable_prosody
        self.enable_linguistic = enable_linguistic
        self.sample_rate = sample_rate
        
        # Initialize feature extractors
        self.prosody_extractor = ProsodyExtractor(sample_rate) if enable_prosody else None
        self.linguistic_extractor = LinguisticFeatureExtractor() if enable_linguistic else None
        
        # Initialize fusion network
        prosody_dim = 25 if enable_prosody else 0
        linguistic_dim = 15 if enable_linguistic else 0
        
        if prosody_dim > 0 or linguistic_dim > 0:
            self.fusion_network = MultimodalFusionNetwork(
                audio_dim=audio_dim,
                prosody_dim=max(prosody_dim, 1),  # Avoid zero dimension
                linguistic_dim=max(linguistic_dim, 1)
            )
        else:
            self.fusion_network = None
        
        logger.info(f"🎭 Initialized multimodal speaker system "
                   f"(prosody={enable_prosody}, linguistic={enable_linguistic})")
    
    def extract_multimodal_features(self, 
                                  audio_embedding: np.ndarray,
                                  audio_waveform: Optional[np.ndarray] = None,
                                  transcription: Optional[str] = None) -> np.ndarray:
        """
        Extract and fuse multimodal features.
        
        Args:
            audio_embedding: Base audio embedding
            audio_waveform: Raw audio for prosodic analysis
            transcription: Text transcription for linguistic analysis
            
        Returns:
            Enhanced multimodal embedding
        """
        try:
            # Start with audio embedding
            features = [torch.from_numpy(audio_embedding).float()]
            
            # Extract prosodic features
            if self.enable_prosody and self.prosody_extractor and audio_waveform is not None:
                prosody_features = self.prosody_extractor.extract_prosodic_features(audio_waveform)
                prosody_vector = np.array(list(prosody_features.values()))
                features.append(torch.from_numpy(prosody_vector).float())
            elif self.enable_prosody:
                # Zero padding if prosody enabled but no audio
                features.append(torch.zeros(25))
            
            # Extract linguistic features
            if self.enable_linguistic and self.linguistic_extractor and transcription:
                linguistic_features = self.linguistic_extractor.extract_linguistic_features(transcription)
                linguistic_vector = np.array(list(linguistic_features.values()))
                features.append(torch.from_numpy(linguistic_vector).float())
            elif self.enable_linguistic:
                # Zero padding if linguistic enabled but no transcription
                features.append(torch.zeros(15))
            
            # Fuse features if fusion network available
            if self.fusion_network and len(features) > 1:
                # Ensure all features have batch dimension
                features = [f.unsqueeze(0) if f.dim() == 1 else f for f in features]
                
                with torch.no_grad():
                    if len(features) == 3:
                        fused = self.fusion_network(features[0], features[1], features[2])
                    else:
                        # Handle cases with missing modalities
                        audio_feat = features[0]
                        prosody_feat = features[1] if len(features) > 1 else torch.zeros(1, 25)
                        linguistic_feat = features[2] if len(features) > 2 else torch.zeros(1, 15)
                        fused = self.fusion_network(audio_feat, prosody_feat, linguistic_feat)
                
                return fused.squeeze(0).numpy()
            else:
                # Return original audio embedding if no fusion
                return audio_embedding
                
        except Exception as e:
            logger.warning(f"⚠️ Multimodal feature extraction failed: {e}")
            return audio_embedding
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get relative importance of different feature modalities."""
        if self.fusion_network is None:
            return {"audio": 1.0}
        
        # Simplified importance based on network weights
        try:
            with torch.no_grad():
                audio_weights = torch.norm(self.fusion_network.audio_encoder[0].weight).item()
                prosody_weights = torch.norm(self.fusion_network.prosody_encoder[0].weight).item() if self.enable_prosody else 0
                linguistic_weights = torch.norm(self.fusion_network.linguistic_encoder[0].weight).item() if self.enable_linguistic else 0
                
                total = audio_weights + prosody_weights + linguistic_weights
                
                return {
                    "audio": audio_weights / total,
                    "prosody": prosody_weights / total,
                    "linguistic": linguistic_weights / total
                }
        except Exception:
            return {"audio": 1.0, "prosody": 0.0, "linguistic": 0.0}

# Factory function
def create_multimodal_system(**kwargs) -> MultimodalSpeakerSystem:
    """Create a multimodal speaker system with custom configuration."""
    return MultimodalSpeakerSystem(**kwargs)