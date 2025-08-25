#!/usr/bin/env python3
"""
Few-Shot Learning Module for Rapid Speaker Enrollment

Implements few-shot learning techniques for quick speaker adaptation:
- Prototypical Networks for speaker classification
- Model-Agnostic Meta-Learning (MAML) for fast adaptation
- Relation Networks for similarity learning
- Memory-Augmented Neural Networks for speaker memory

Based on research from:
- "Prototypical Networks for Few-shot Learning" (Snell et al., 2017)
- "Model-Agnostic Meta-Learning" (Finn et al., 2017)
- "Learning to Compare: Relation Network" (Sung et al., 2018)
- "Few-Shot Speaker Recognition" (Zhang et al., 2020)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network for few-shot speaker recognition.
    
    Learns to classify speakers by computing distances to prototype
    representations in embedding space.
    """
    
    def __init__(self, embedding_dim: int = 192, hidden_dim: int = 256):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Embedding network
        self.embedding_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through embedding network."""
        return self.embedding_net(x)
    
    def compute_prototypes(self, support_embeddings: torch.Tensor, support_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute prototype representations for each class.
        
        Args:
            support_embeddings: Support set embeddings [n_support, embedding_dim]
            support_labels: Support set labels [n_support]
            
        Returns:
            Prototypes for each class [n_classes, embedding_dim]
        """
        n_classes = len(torch.unique(support_labels))
        prototypes = torch.zeros(n_classes, self.embedding_dim)
        
        for i, class_id in enumerate(torch.unique(support_labels)):
            class_mask = support_labels == class_id
            class_embeddings = support_embeddings[class_mask]
            prototypes[i] = class_embeddings.mean(dim=0)
        
        return prototypes
    
    def classify(self, query_embeddings: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """
        Classify query embeddings using prototypes.
        
        Args:
            query_embeddings: Query embeddings [n_query, embedding_dim]
            prototypes: Class prototypes [n_classes, embedding_dim]
            
        Returns:
            Classification logits [n_query, n_classes]
        """
        # Compute distances to prototypes
        distances = torch.cdist(query_embeddings, prototypes, p=2)
        
        # Convert distances to logits (negative distance)
        logits = -distances
        
        return logits

class RelationNetwork(nn.Module):
    """
    Relation Network for few-shot speaker recognition.
    
    Learns to compute similarity scores between query and support embeddings
    using a learned relation function.
    """
    
    def __init__(self, embedding_dim: int = 192, relation_dim: int = 128):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.relation_dim = relation_dim
        
        # Feature embedding network
        self.feature_net = nn.Sequential(
            nn.Linear(embedding_dim, relation_dim),
            nn.ReLU(),
            nn.BatchNorm1d(relation_dim),
            nn.Dropout(0.2),
            nn.Linear(relation_dim, relation_dim),
            nn.ReLU(),
            nn.BatchNorm1d(relation_dim)
        )
        
        # Relation network
        self.relation_net = nn.Sequential(
            nn.Linear(relation_dim * 2, relation_dim),
            nn.ReLU(),
            nn.BatchNorm1d(relation_dim),
            nn.Dropout(0.3),
            nn.Linear(relation_dim, relation_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(relation_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, query: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        """
        Compute relation scores between query and support embeddings.
        
        Args:
            query: Query embeddings [n_query, embedding_dim]
            support: Support embeddings [n_support, embedding_dim]
            
        Returns:
            Relation scores [n_query, n_support]
        """
        # Extract features
        query_features = self.feature_net(query)  # [n_query, relation_dim]
        support_features = self.feature_net(support)  # [n_support, relation_dim]
        
        # Compute pairwise relations
        n_query = query_features.size(0)
        n_support = support_features.size(0)
        
        # Expand for pairwise computation
        query_expanded = query_features.unsqueeze(1).expand(n_query, n_support, -1)
        support_expanded = support_features.unsqueeze(0).expand(n_query, n_support, -1)
        
        # Concatenate query and support features
        relation_pairs = torch.cat([query_expanded, support_expanded], dim=2)
        relation_pairs = relation_pairs.view(-1, self.relation_dim * 2)
        
        # Compute relation scores
        relation_scores = self.relation_net(relation_pairs)
        relation_scores = relation_scores.view(n_query, n_support)
        
        return relation_scores

class MAMLSpeakerAdapter(nn.Module):
    """
    Model-Agnostic Meta-Learning (MAML) for speaker adaptation.
    
    Learns initialization parameters that can be quickly adapted
    to new speakers with few examples.
    """
    
    def __init__(self, embedding_dim: int = 192, hidden_dim: int = 256):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Adaptation network
        self.adaptation_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Meta-learning parameters
        self.meta_lr = 0.01
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through adaptation network."""
        return self.adaptation_net(x)
    
    def adapt(self, support_embeddings: torch.Tensor, support_labels: torch.Tensor, 
              n_steps: int = 5) -> nn.Module:
        """
        Adapt the model to new speakers using support examples.
        
        Args:
            support_embeddings: Support set embeddings
            support_labels: Support set labels
            n_steps: Number of adaptation steps
            
        Returns:
            Adapted model
        """
        # Create a copy of the model for adaptation
        adapted_model = type(self)(self.embedding_dim, self.hidden_dim)
        adapted_model.load_state_dict(self.state_dict())
        
        # Adaptation loop
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.meta_lr)
        
        for step in range(n_steps):
            # Forward pass
            adapted_embeddings = adapted_model(support_embeddings)
            
            # Compute adaptation loss (reconstruction + classification)
            reconstruction_loss = F.mse_loss(adapted_embeddings, support_embeddings)
            
            # Simple classification loss using prototypes
            prototypes = self._compute_prototypes(adapted_embeddings, support_labels)
            classification_loss = self._compute_classification_loss(adapted_embeddings, support_labels, prototypes)
            
            total_loss = reconstruction_loss + classification_loss
            
            # Backward pass and update
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        return adapted_model
    
    def _compute_prototypes(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute class prototypes."""
        unique_labels = torch.unique(labels)
        prototypes = torch.zeros(len(unique_labels), self.embedding_dim)
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            prototypes[i] = embeddings[mask].mean(dim=0)
        
        return prototypes
    
    def _compute_classification_loss(self, embeddings: torch.Tensor, labels: torch.Tensor, 
                                   prototypes: torch.Tensor) -> torch.Tensor:
        """Compute classification loss using prototypes."""
        distances = torch.cdist(embeddings, prototypes, p=2)
        logits = -distances
        return F.cross_entropy(logits, labels)

class MemoryAugmentedSpeakerNet(nn.Module):
    """
    Memory-Augmented Neural Network for speaker recognition.
    
    Maintains an external memory of speaker representations
    that can be quickly accessed and updated.
    """
    
    def __init__(self, embedding_dim: int = 192, memory_size: int = 1000, memory_dim: int = 256):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        
        # Memory matrix
        self.memory = nn.Parameter(torch.randn(memory_size, memory_dim))
        
        # Controller network
        self.controller = nn.Sequential(
            nn.Linear(embedding_dim, memory_dim),
            nn.ReLU(),
            nn.BatchNorm1d(memory_dim),
            nn.Dropout(0.2),
            nn.Linear(memory_dim, memory_dim)
        )
        
        # Output network
        self.output_net = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.ReLU(),
            nn.BatchNorm1d(memory_dim),
            nn.Dropout(0.2),
            nn.Linear(memory_dim, embedding_dim)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with memory access.
        
        Args:
            x: Input embeddings [batch_size, embedding_dim]
            
        Returns:
            Output embeddings and attention weights
        """
        batch_size = x.size(0)
        
        # Generate query from input
        query = self.controller(x)  # [batch_size, memory_dim]
        
        # Compute attention weights over memory
        attention_weights = F.softmax(
            torch.matmul(query, self.memory.t()), dim=1
        )  # [batch_size, memory_size]
        
        # Read from memory
        memory_output = torch.matmul(attention_weights, self.memory)  # [batch_size, memory_dim]
        
        # Combine query and memory output
        combined = torch.cat([query, memory_output], dim=1)  # [batch_size, memory_dim * 2]
        
        # Generate output
        output = self.output_net(combined)  # [batch_size, embedding_dim]
        
        return output, attention_weights
    
    def update_memory(self, embeddings: torch.Tensor, speaker_ids: torch.Tensor):
        """
        Update memory with new speaker embeddings.
        
        Args:
            embeddings: New speaker embeddings
            speaker_ids: Corresponding speaker IDs
        """
        with torch.no_grad():
            queries = self.controller(embeddings)
            
            # Find least used memory slots for updates
            for i, (query, speaker_id) in enumerate(zip(queries, speaker_ids)):
                # Simple update strategy: replace least similar slot
                similarities = F.cosine_similarity(query.unsqueeze(0), self.memory, dim=1)
                min_idx = torch.argmin(similarities)
                
                # Update memory slot with exponential moving average
                alpha = 0.1
                self.memory[min_idx] = (1 - alpha) * self.memory[min_idx] + alpha * query

class FewShotSpeakerSystem:
    """
    Complete few-shot learning system for speaker recognition.
    
    Integrates multiple few-shot learning approaches for rapid
    speaker enrollment and adaptation.
    """
    
    def __init__(self, 
                 embedding_dim: int = 192,
                 method: str = "prototypical",
                 enable_memory: bool = True):
        """
        Initialize few-shot speaker system.
        
        Args:
            embedding_dim: Dimension of speaker embeddings
            method: Few-shot method ("prototypical", "relation", "maml", "auto")
            enable_memory: Enable memory-augmented learning
        """
        self.embedding_dim = embedding_dim
        self.method = method
        self.enable_memory = enable_memory
        
        # Initialize models
        self.prototypical_net = None
        self.relation_net = None
        self.maml_adapter = None
        self.memory_net = None
        
        self._initialize_models()
        
        # Speaker enrollment data
        self.enrolled_speakers = {}
        self.speaker_prototypes = {}
        
        # Statistics
        self.stats = {
            "enrollments": 0,
            "adaptations": 0,
            "few_shot_accuracy": 0.0,
            "memory_usage": 0
        }
        
        logger.info(f"🎯 Initialized few-shot speaker system with {method} method")
    
    def _initialize_models(self):
        """Initialize few-shot learning models."""
        try:
            if self.method in ["prototypical", "auto"]:
                self.prototypical_net = PrototypicalNetwork(self.embedding_dim)
                
            if self.method in ["relation", "auto"]:
                self.relation_net = RelationNetwork(self.embedding_dim)
                
            if self.method in ["maml", "auto"]:
                self.maml_adapter = MAMLSpeakerAdapter(self.embedding_dim)
                
            if self.enable_memory:
                self.memory_net = MemoryAugmentedSpeakerNet(self.embedding_dim)
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize few-shot models: {e}")
    
    def enroll_speaker(self, 
                      speaker_id: str, 
                      embeddings: List[np.ndarray],
                      min_shots: int = 3) -> bool:
        """
        Enroll a new speaker with few-shot learning.
        
        Args:
            speaker_id: Unique speaker identifier
            embeddings: List of speaker embeddings (few examples)
            min_shots: Minimum number of examples required
            
        Returns:
            Success status
        """
        try:
            if len(embeddings) < min_shots:
                logger.warning(f"⚠️ Insufficient examples for speaker {speaker_id}: {len(embeddings)} < {min_shots}")
                return False
            
            # Convert to tensor
            embedding_tensor = torch.from_numpy(np.array(embeddings)).float()
            
            # Compute prototype for the speaker
            prototype = embedding_tensor.mean(dim=0)
            
            # Store enrollment data
            self.enrolled_speakers[speaker_id] = {
                'embeddings': embeddings,
                'prototype': prototype.numpy(),
                'enrollment_time': torch.tensor(0.0),  # Placeholder
                'adaptation_count': 0
            }
            
            self.speaker_prototypes[speaker_id] = prototype
            
            # Update memory if enabled
            if self.enable_memory and self.memory_net is not None:
                speaker_labels = torch.full((len(embeddings),), hash(speaker_id) % 1000, dtype=torch.long)
                self.memory_net.update_memory(embedding_tensor, speaker_labels)
            
            self.stats["enrollments"] += 1
            logger.info(f"✅ Enrolled speaker {speaker_id} with {len(embeddings)} examples")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to enroll speaker {speaker_id}: {e}")
            return False
    
    def identify_speaker(self, 
                        query_embedding: np.ndarray,
                        top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Identify speaker using few-shot learning.
        
        Args:
            query_embedding: Query embedding to identify
            top_k: Number of top candidates to return
            
        Returns:
            List of (speaker_id, confidence) tuples
        """
        try:
            if not self.enrolled_speakers:
                return []
            
            query_tensor = torch.from_numpy(query_embedding).float().unsqueeze(0)
            
            # Method selection
            if self.method == "prototypical" or (self.method == "auto" and self.prototypical_net):
                return self._identify_prototypical(query_tensor, top_k)
            elif self.method == "relation" or (self.method == "auto" and self.relation_net):
                return self._identify_relation(query_tensor, top_k)
            elif self.method == "maml" or (self.method == "auto" and self.maml_adapter):
                return self._identify_maml(query_tensor, top_k)
            else:
                # Fallback to simple prototype matching
                return self._identify_simple(query_embedding, top_k)
                
        except Exception as e:
            logger.error(f"❌ Speaker identification failed: {e}")
            return []
    
    def _identify_prototypical(self, query_tensor: torch.Tensor, top_k: int) -> List[Tuple[str, float]]:
        """Identify using prototypical networks."""
        if self.prototypical_net is None:
            return []
        
        # Stack all prototypes
        speaker_ids = list(self.speaker_prototypes.keys())
        prototypes = torch.stack([self.speaker_prototypes[sid] for sid in speaker_ids])
        
        # Compute distances
        with torch.no_grad():
            query_processed = self.prototypical_net(query_tensor)
            distances = torch.cdist(query_processed, prototypes, p=2).squeeze(0)
            
            # Convert to similarities (negative distances)
            similarities = -distances
            
            # Get top-k
            top_indices = torch.topk(similarities, min(top_k, len(speaker_ids))).indices
            
            results = []
            for idx in top_indices:
                speaker_id = speaker_ids[idx]
                confidence = torch.sigmoid(similarities[idx]).item()
                results.append((speaker_id, confidence))
            
            return results
    
    def _identify_relation(self, query_tensor: torch.Tensor, top_k: int) -> List[Tuple[str, float]]:
        """Identify using relation networks."""
        if self.relation_net is None:
            return []
        
        results = []
        
        with torch.no_grad():
            for speaker_id, data in self.enrolled_speakers.items():
                # Use stored embeddings as support set
                support_embeddings = torch.from_numpy(np.array(data['embeddings'])).float()
                
                # Compute relation scores
                relation_scores = self.relation_net(query_tensor, support_embeddings)
                avg_score = relation_scores.mean().item()
                
                results.append((speaker_id, avg_score))
        
        # Sort by confidence and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _identify_maml(self, query_tensor: torch.Tensor, top_k: int) -> List[Tuple[str, float]]:
        """Identify using MAML adaptation."""
        if self.maml_adapter is None:
            return []
        
        # Simplified MAML identification
        return self._identify_simple(query_tensor.squeeze(0).numpy(), top_k)
    
    def _identify_simple(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """Simple prototype-based identification."""
        results = []
        
        for speaker_id, data in self.enrolled_speakers.items():
            prototype = data['prototype']
            
            # Cosine similarity
            similarity = np.dot(query_embedding, prototype) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(prototype)
            )
            
            results.append((speaker_id, float(similarity)))
        
        # Sort by similarity and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_enrollment_status(self) -> Dict:
        """Get comprehensive enrollment status."""
        return {
            "total_speakers": len(self.enrolled_speakers),
            "method": self.method,
            "memory_enabled": self.enable_memory,
            "statistics": self.stats.copy(),
            "speakers": {
                speaker_id: {
                    "num_embeddings": len(data['embeddings']),
                    "adaptation_count": data['adaptation_count']
                }
                for speaker_id, data in self.enrolled_speakers.items()
            }
        }

# Factory function
def create_few_shot_system(**kwargs) -> FewShotSpeakerSystem:
    """Create a few-shot speaker system with custom configuration."""
    return FewShotSpeakerSystem(**kwargs)