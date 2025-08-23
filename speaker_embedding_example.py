#!/usr/bin/env python3
"""
Example of proper speaker embedding implementation for consistent speaker identification.
This shows how to extract, store, and match speaker embeddings.
"""

import numpy as np
import torch
import torchaudio
from pyannote.audio import Model
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class SpeakerEmbeddingSystem:
    """
    A complete speaker embedding system for consistent speaker identification.
    """
    
    def __init__(self, model_name: str = "pyannote/embedding", similarity_threshold: float = 0.75):
        """
        Initialize the speaker embedding system.
        
        Args:
            model_name: HuggingFace model for speaker embeddings
            similarity_threshold: Minimum similarity to consider speakers the same
        """
        self.model = Model.from_pretrained(model_name)
        self.similarity_threshold = similarity_threshold
        self.sample_rate = 16000
        
        # Speaker database
        self.speaker_embeddings: Dict[str, List[np.ndarray]] = {}  # speaker_id -> embeddings
        self.speaker_metadata: Dict[str, dict] = {}  # speaker_id -> metadata
        self.next_speaker_id = 1
        
    def extract_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> Optional[np.ndarray]:
        """
        Extract speaker embedding from audio segment.
        
        Args:
            audio: Audio waveform (1D numpy array)
            sample_rate: Sample rate of audio
            
        Returns:
            Speaker embedding vector or None if extraction fails
        """
        try:
            # Ensure audio is the right format
            if len(audio.shape) == 1:
                audio = audio.reshape(1, -1)
            
            # Convert to torch tensor
            waveform = torch.from_numpy(audio).float()
            
            # Resample if needed
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)
            
            # Ensure minimum duration (2 seconds for good embeddings)
            min_samples = self.sample_rate * 2
            if waveform.shape[1] < min_samples:
                # Pad with zeros if too short
                padding = min_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model(waveform)
                if hasattr(embedding, 'data'):
                    embedding = embedding.data
                embedding = embedding.cpu().numpy().flatten()
            
            return embedding
            
        except Exception as e:
            print(f"Failed to extract embedding: {e}")
            return None
    
    def find_matching_speaker(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Find the best matching speaker for an embedding.
        
        Args:
            embedding: Speaker embedding to match
            
        Returns:
            Tuple of (speaker_id, similarity_score) or (None, 0.0) if no match
        """
        if embedding is None:
            return None, 0.0
        
        best_speaker = None
        best_similarity = 0.0
        
        for speaker_id, embeddings in self.speaker_embeddings.items():
            if not embeddings:
                continue
            
            # Calculate average embedding for this speaker
            avg_embedding = np.mean(embeddings, axis=0)
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embedding], [avg_embedding])[0][0]
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_speaker = speaker_id
        
        # Only return match if above threshold
        if best_similarity >= self.similarity_threshold:
            return best_speaker, best_similarity
        else:
            return None, best_similarity
    
    def register_speaker(self, embedding: np.ndarray, speaker_name: Optional[str] = None) -> str:
        """
        Register a new speaker or add embedding to existing speaker.
        
        Args:
            embedding: Speaker embedding
            speaker_name: Optional human-readable name
            
        Returns:
            Speaker ID (existing or new)
        """
        if embedding is None:
            return None
        
        # Try to find existing speaker
        existing_speaker, similarity = self.find_matching_speaker(embedding)
        
        if existing_speaker:
            # Add to existing speaker
            self.speaker_embeddings[existing_speaker].append(embedding)
            print(f"Added embedding to existing speaker {existing_speaker} (similarity: {similarity:.3f})")
            return existing_speaker
        else:
            # Create new speaker
            speaker_id = f"speaker_{self.next_speaker_id}"
            self.next_speaker_id += 1
            
            self.speaker_embeddings[speaker_id] = [embedding]
            self.speaker_metadata[speaker_id] = {
                'name': speaker_name or speaker_id,
                'created_at': str(np.datetime64('now')),
                'embedding_count': 1
            }
            
            print(f"Created new speaker {speaker_id}")
            return speaker_id
    
    def identify_speaker(self, audio: np.ndarray, sample_rate: int = 16000) -> Tuple[str, float]:
        """
        Identify speaker from audio segment.
        
        Args:
            audio: Audio waveform
            sample_rate: Sample rate
            
        Returns:
            Tuple of (speaker_id, confidence)
        """
        # Extract embedding
        embedding = self.extract_embedding(audio, sample_rate)
        if embedding is None:
            return "unknown", 0.0
        
        # Find matching speaker
        speaker_id, similarity = self.find_matching_speaker(embedding)
        
        if speaker_id:
            # Add this embedding to improve the speaker model
            self.speaker_embeddings[speaker_id].append(embedding)
            self.speaker_metadata[speaker_id]['embedding_count'] += 1
            return speaker_id, similarity
        else:
            # Register as new speaker
            new_speaker_id = self.register_speaker(embedding)
            return new_speaker_id, 1.0  # Perfect match for new speaker
    
    def save_speaker_database(self, filepath: str):
        """Save speaker database to disk."""
        data = {
            'embeddings': {k: [emb.tolist() for emb in v] for k, v in self.speaker_embeddings.items()},
            'metadata': self.speaker_metadata,
            'next_speaker_id': self.next_speaker_id,
            'similarity_threshold': self.similarity_threshold
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved speaker database to {filepath}")
    
    def load_speaker_database(self, filepath: str):
        """Load speaker database from disk."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.speaker_embeddings = {
                k: [np.array(emb) for emb in v] 
                for k, v in data['embeddings'].items()
            }
            self.speaker_metadata = data['metadata']
            self.next_speaker_id = data['next_speaker_id']
            self.similarity_threshold = data['similarity_threshold']
            
            print(f"Loaded speaker database from {filepath}")
            print(f"Found {len(self.speaker_embeddings)} speakers")
            
        except Exception as e:
            print(f"Failed to load speaker database: {e}")


# Example usage
if __name__ == "__main__":
    # Initialize the system
    embedding_system = SpeakerEmbeddingSystem()
    
    # Example: Process audio segments
    # audio1 = np.random.randn(32000)  # 2 seconds of audio at 16kHz
    # audio2 = np.random.randn(48000)  # 3 seconds of audio
    
    # # Identify speakers
    # speaker1, conf1 = embedding_system.identify_speaker(audio1)
    # speaker2, conf2 = embedding_system.identify_speaker(audio2)
    
    # print(f"Audio 1: {speaker1} (confidence: {conf1:.3f})")
    # print(f"Audio 2: {speaker2} (confidence: {conf2:.3f})")
    
    # # Save the database
    # embedding_system.save_speaker_database("speakers.json")
    
    print("Speaker embedding system ready!")
