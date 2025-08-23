#!/usr/bin/env python3
"""
Simple demonstration of how speaker embeddings work for identification.
This shows the core concept without the complexity.
"""

import numpy as np
import torch
from pyannote.audio import Model
from sklearn.metrics.pairwise import cosine_similarity
import json

class SimpleSpeakerIdentifier:
    def __init__(self):
        # Load pre-trained speaker embedding model
        self.model = Model.from_pretrained("pyannote/embedding")
        self.speaker_database = {}  # speaker_name -> list of embeddings
        
    def extract_embedding(self, audio_waveform):
        """
        Extract a speaker embedding from audio.
        
        Args:
            audio_waveform: numpy array of audio samples (16kHz)
        Returns:
            192-dimensional embedding vector
        """
        # Convert to torch tensor
        if len(audio_waveform.shape) == 1:
            audio_waveform = audio_waveform.reshape(1, -1)
        
        waveform = torch.from_numpy(audio_waveform).float()
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.model(waveform)
            return embedding.cpu().numpy().flatten()
    
    def register_speaker(self, speaker_name, audio_samples):
        """
        Register a new speaker with multiple audio samples.
        
        Args:
            speaker_name: Human-readable name
            audio_samples: List of audio waveforms from this speaker
        """
        embeddings = []
        for audio in audio_samples:
            embedding = self.extract_embedding(audio)
            embeddings.append(embedding)
        
        self.speaker_database[speaker_name] = embeddings
        print(f"✅ Registered {speaker_name} with {len(embeddings)} voice samples")
    
    def identify_speaker(self, audio_waveform, threshold=0.75):
        """
        Identify which registered speaker this audio belongs to.
        
        Args:
            audio_waveform: Audio to identify
            threshold: Minimum similarity to consider a match
            
        Returns:
            (speaker_name, confidence) or ("unknown", 0.0)
        """
        # Extract embedding from unknown audio
        unknown_embedding = self.extract_embedding(audio_waveform)
        
        best_match = None
        best_similarity = 0.0
        
        # Compare against all registered speakers
        for speaker_name, embeddings in self.speaker_database.items():
            # Calculate average embedding for this speaker
            avg_embedding = np.mean(embeddings, axis=0)
            
            # Calculate similarity
            similarity = cosine_similarity([unknown_embedding], [avg_embedding])[0][0]
            
            print(f"Similarity to {speaker_name}: {similarity:.3f}")
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = speaker_name
        
        # Return match if above threshold
        if best_similarity >= threshold:
            return best_match, best_similarity
        else:
            return "unknown", best_similarity
    
    def save_database(self, filename):
        """Save speaker database to file."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_db = {}
        for name, embeddings in self.speaker_database.items():
            serializable_db[name] = [emb.tolist() for emb in embeddings]
        
        with open(filename, 'w') as f:
            json.dump(serializable_db, f, indent=2)
        print(f"💾 Saved speaker database to {filename}")
    
    def load_database(self, filename):
        """Load speaker database from file."""
        try:
            with open(filename, 'r') as f:
                serializable_db = json.load(f)
            
            # Convert lists back to numpy arrays
            self.speaker_database = {}
            for name, embeddings in serializable_db.items():
                self.speaker_database[name] = [np.array(emb) for emb in embeddings]
            
            print(f"📂 Loaded {len(self.speaker_database)} speakers from {filename}")
        except FileNotFoundError:
            print(f"❌ Database file {filename} not found")


# Example usage demonstrating the concept
if __name__ == "__main__":
    identifier = SimpleSpeakerIdentifier()
    
    print("🎤 Speaker Embedding Demo")
    print("=" * 50)
    
    # Simulate registering speakers with voice samples
    # In real usage, these would be actual audio recordings
    
    # Example: Register "Alice" with 3 voice samples
    alice_samples = [
        np.random.randn(32000),  # 2 seconds at 16kHz
        np.random.randn(48000),  # 3 seconds
        np.random.randn(24000),  # 1.5 seconds
    ]
    identifier.register_speaker("Alice", alice_samples)
    
    # Example: Register "Bob" with 2 voice samples  
    bob_samples = [
        np.random.randn(40000),  # 2.5 seconds
        np.random.randn(32000),  # 2 seconds
    ]
    identifier.register_speaker("Bob", bob_samples)
    
    # Test identification
    print("\n🔍 Testing Speaker Identification:")
    test_audio = np.random.randn(32000)  # Unknown speaker
    speaker, confidence = identifier.identify_speaker(test_audio)
    print(f"Identified as: {speaker} (confidence: {confidence:.3f})")
    
    # Save the database
    identifier.save_database("speaker_profiles.json")
    
    print("\n✨ This is how you build persistent speaker identification!")
    print("The embeddings capture unique voice characteristics that remain")
    print("consistent across different recordings of the same person.")
