import asyncio
import re
import threading
import numpy as np
import logging
import time
from queue import SimpleQueue, Empty
from typing import Tuple, Any, List

from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.inference import StreamingInference
from diart.sources import AudioSource
from whisperlivekit.timed_objects import SpeakerSegment
from diart.sources import MicrophoneAudioSource
from rx.core import Observer
from pyannote.core import Annotation
import diart.models as m

logger = logging.getLogger(__name__)

def extract_number(s: str) -> int:
    m = re.search(r'\d+', s)
    return int(m.group()) if m else None

class DiarizationObserver(Observer):
    """Observer that logs all data emitted by the diarization pipeline and stores speaker segments."""
    
    def __init__(self):
        self.speaker_segments = []
        self.processed_time = 0
        self.segment_lock = threading.Lock()
        self.global_time_offset = 0.0
    
    def on_next(self, value: Tuple[Annotation, Any]):
        annotation, audio = value
        
        logger.debug("\n--- New Diarization Result ---")
        
        duration = audio.extent.end - audio.extent.start
        logger.debug(f"Audio segment: {audio.extent.start:.2f}s - {audio.extent.end:.2f}s (duration: {duration:.2f}s)")
        logger.debug(f"Audio shape: {audio.data.shape}")
        
        with self.segment_lock:
            if audio.extent.end > self.processed_time:
                self.processed_time = audio.extent.end            
            if annotation and len(annotation._labels) > 0:
                logger.debug("\nSpeaker segments:")
                for speaker, label in annotation._labels.items():
                    for start, end in zip(label.segments_boundaries_[:-1], label.segments_boundaries_[1:]):
                        print(f"  {speaker}: {start:.2f}s-{end:.2f}s")
                        self.speaker_segments.append(SpeakerSegment(
                            speaker=speaker,
                            start=start + self.global_time_offset,
                            end=end + self.global_time_offset
                        ))
            else:
                logger.debug("\nNo speakers detected in this segment")
                
    def get_segments(self) -> List[SpeakerSegment]:
        """Get a copy of the current speaker segments."""
        with self.segment_lock:
            return self.speaker_segments.copy()
    
    def clear_old_segments(self, older_than: float = 30.0):
        """Clear segments older than the specified time."""
        with self.segment_lock:
            current_time = self.processed_time
            self.speaker_segments = [
                segment for segment in self.speaker_segments 
                if current_time - segment.end < older_than
            ]
    
    def on_error(self, error):
        """Handle an error in the stream."""
        logger.debug(f"Error in diarization stream: {error}")
        
    def on_completed(self):
        """Handle the completion of the stream."""
        logger.debug("Diarization stream completed")


class WebSocketAudioSource(AudioSource):
    """
    Buffers incoming audio and releases it in fixed-size chunks at regular intervals.
    """
    def __init__(self, uri: str = "websocket", sample_rate: int = 16000, block_duration: float = 0.5):
        super().__init__(uri, sample_rate)
        self.block_duration = block_duration
        self.block_size = int(np.rint(block_duration * sample_rate))
        self._queue = SimpleQueue()
        self._buffer = np.array([], dtype=np.float32)
        self._buffer_lock = threading.Lock()
        self._closed = False
        self._close_event = threading.Event()
        self._processing_thread = None
        self._last_chunk_time = time.time()

    def read(self):
        """Start processing buffered audio and emit fixed-size chunks."""
        self._processing_thread = threading.Thread(target=self._process_chunks)
        self._processing_thread.daemon = True
        self._processing_thread.start()
        
        self._close_event.wait()
        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)

    def _process_chunks(self):
        """Process audio from queue and emit fixed-size chunks at regular intervals."""
        while not self._closed:
            try:
                audio_chunk = self._queue.get(timeout=0.1)
                
                with self._buffer_lock:
                    self._buffer = np.concatenate([self._buffer, audio_chunk])
                    
                    while len(self._buffer) >= self.block_size:
                        chunk = self._buffer[:self.block_size]
                        self._buffer = self._buffer[self.block_size:]
                        
                        current_time = time.time()
                        time_since_last = current_time - self._last_chunk_time
                        if time_since_last < self.block_duration:
                            time.sleep(self.block_duration - time_since_last)
                        
                        chunk_reshaped = chunk.reshape(1, -1)
                        self.stream.on_next(chunk_reshaped)
                        self._last_chunk_time = time.time()
                        
            except Empty:
                with self._buffer_lock:
                    if len(self._buffer) > 0 and time.time() - self._last_chunk_time > self.block_duration:
                        padded_chunk = np.zeros(self.block_size, dtype=np.float32)
                        padded_chunk[:len(self._buffer)] = self._buffer
                        self._buffer = np.array([], dtype=np.float32)
                        
                        chunk_reshaped = padded_chunk.reshape(1, -1)
                        self.stream.on_next(chunk_reshaped)
                        self._last_chunk_time = time.time()
            except Exception as e:
                logger.error(f"Error in audio processing thread: {e}")
                self.stream.on_error(e)
                break
        
        with self._buffer_lock:
            if len(self._buffer) > 0:
                padded_chunk = np.zeros(self.block_size, dtype=np.float32)
                padded_chunk[:len(self._buffer)] = self._buffer
                chunk_reshaped = padded_chunk.reshape(1, -1)
                self.stream.on_next(chunk_reshaped)
        
        self.stream.on_completed()

    def close(self):
        if not self._closed:
            self._closed = True
            self._close_event.set()

    def push_audio(self, chunk: np.ndarray):
        """Add audio chunk to the processing queue."""
        if not self._closed:
            if chunk.ndim > 1:
                chunk = chunk.flatten()
            self._queue.put(chunk)
            logger.debug(f'Added chunk to queue with {len(chunk)} samples')


class DiartDiarization:
    def __init__(self, sample_rate: int = 16000, config : SpeakerDiarizationConfig = None, use_microphone: bool = False, block_duration: float = 1.5, segmentation_model_name: str = "pyannote/segmentation-3.0", embedding_model_name: str = "pyannote/embedding"):
        segmentation_model = m.SegmentationModel.from_pretrained(segmentation_model_name)
        embedding_model = m.EmbeddingModel.from_pretrained(embedding_model_name)
        
        if config is None:
            config = SpeakerDiarizationConfig(
                segmentation=segmentation_model,
                embedding=embedding_model,
            )
        
        self.pipeline = SpeakerDiarization(config=config)        
        self.observer = DiarizationObserver()
        self.lag_diart = None
        
        if use_microphone:
            self.source = MicrophoneAudioSource(block_duration=block_duration)
            self.custom_source = None
        else:
            self.custom_source = WebSocketAudioSource(
                uri="websocket_source", 
                sample_rate=sample_rate,
                block_duration=block_duration
            )
            self.source = self.custom_source
            
        self.inference = StreamingInference(
            pipeline=self.pipeline,
            source=self.source,
            do_plot=False,
            show_progress=False,
        )
        self.inference.attach_observers(self.observer)
        asyncio.get_event_loop().run_in_executor(None, self.inference)

    def insert_silence(self, silence_duration):
        self.observer.global_time_offset += silence_duration

    async def diarize(self, pcm_array: np.ndarray):
        """
        Process audio data for diarization.
        Only used when working with WebSocketAudioSource.
        """
        if self.custom_source:
            self.custom_source.push_audio(pcm_array)            
        # self.observer.clear_old_segments()        

    def close(self):
        """Close the audio source."""
        if self.custom_source:
            self.custom_source.close()

    def assign_speakers_to_tokens(self, tokens: list, use_punctuation_split: bool = False) -> float:
        """
        Assign speakers to tokens based on timing overlap with speaker segments.
        Uses the segments collected by the observer.
        
        If use_punctuation_split is True, uses punctuation marks to refine speaker boundaries.
        """
        # Periodically clear old segments to avoid stale assignments and drift
        self.observer.clear_old_segments(older_than=30.0)
        segments = self.observer.get_segments()
        
        # Debug logging
        logger.info(f"assign_speakers_to_tokens called with {len(tokens)} tokens")
        logger.info(f"Available segments: {len(segments)}")
        for i, seg in enumerate(segments[:5]):  # Show first 5 segments
            logger.info(f"  Segment {i}: {seg.speaker} [{seg.start:.2f}-{seg.end:.2f}]")
        
        if not self.lag_diart and segments and tokens:
            self.lag_diart = segments[0].start - tokens[0].start

        if not use_punctuation_split:
            # Step 1: continuously refine lag estimate using EMA when confident
            self._update_lag_ema(tokens, segments)

            # Step 2: group tokens into utterances by punctuation and inter-token gap
            utterances = self._group_tokens_into_utterances(tokens)

            # Step 3: assign a single speaker per utterance using majority overlap
            self._assign_utterances_by_overlap(tokens, utterances, segments)

            # Step 4: minimal cleanup of obvious single-token errors
            self._fix_short_segments(tokens)

            # Log final assignments
            for token in tokens:
                logger.info(f"Final assignment: '{token.text}' [{token.start:.2f}-{token.end:.2f}] -> Speaker {token.speaker}")
        else:
            tokens = add_speaker_to_tokens(segments, tokens)
        return tokens

    def _update_lag_ema(self, tokens, segments, alpha: float = 0.12, min_overlap_ratio: float = 0.5):
        """
        Update diarization lag using an exponential moving average based on
        tokens that confidently overlap a single diarization segment.
        """
        if not tokens or not segments:
            return
        current_lag = self.lag_diart or 0.0
        updates = 0
        for token in tokens:
            token_start = token.start + current_lag
            token_end = token.end + current_lag
            token_dur = max(1e-6, token_end - token_start)

            best_seg = None
            best_overlap = 0.0
            second_overlap = 0.0
            for seg in segments:
                overlap_start = max(seg.start, token_start)
                overlap_end = min(seg.end, token_end)
                overlap = max(0.0, overlap_end - overlap_start)
                if overlap > best_overlap:
                    second_overlap = best_overlap
                    best_overlap = overlap
                    best_seg = seg
                elif overlap > second_overlap:
                    second_overlap = overlap

            if best_seg is None:
                continue

            # Require clean dominance to avoid noisy updates
            if best_overlap / token_dur >= min_overlap_ratio and best_overlap > second_overlap * 1.5:
                token_mid = (token.start + token.end) / 2.0
                seg_mid = (best_seg.start + best_seg.end) / 2.0
                delta = seg_mid - token_mid
                current_lag = (1 - alpha) * current_lag + alpha * delta
                updates += 1

        if updates > 0:
            self.lag_diart = current_lag

    def _group_tokens_into_utterances(self, tokens, max_gap: float = 0.35):
        """
        Group tokens into utterances using punctuation boundaries and time gaps.
        Returns a list of (start_index, end_index) inclusive utterance spans.
        """
        if not tokens:
            return []
        punctuation_marks = {'.', '!', '?'}
        utterances = []
        start_idx = 0
        last_end = tokens[0].end
        for i in range(1, len(tokens)):
            gap = tokens[i].start - last_end
            last_end = tokens[i].end
            is_boundary = tokens[i-1].text.strip() in punctuation_marks or gap > max_gap
            if is_boundary:
                utterances.append((start_idx, i - 1))
                start_idx = i
        utterances.append((start_idx, len(tokens) - 1))
        return utterances

    def _assign_utterances_by_overlap(self, tokens, utterances, segments,
                                      hysteresis_margin: float = 0.2):
        """
        Assign a single speaker per utterance using simple timing overlap.
        No embeddings - just pure timing-based assignment.
        """
        if not segments or not tokens or not utterances:
            return
        prev_speaker = None
        lag = self.lag_diart or 0.0

        for (s_idx, e_idx) in utterances:
            # Calculate timing-based overlaps for all speakers
            speaker_to_overlap = {}
            
            for i in range(s_idx, e_idx + 1):
                t = tokens[i]
                t_start = t.start + lag
                t_end = t.end + lag
                
                for seg in segments:
                    overlap_start = max(seg.start, t_start)
                    overlap_end = min(seg.end, t_end)
                    overlap = max(0, overlap_end - overlap_start)
                    
                    if overlap > 0:
                        speaker_id = extract_number(seg.speaker)
                        speaker_to_overlap[speaker_id] = speaker_to_overlap.get(speaker_id, 0) + overlap
            
            if speaker_to_overlap:
                # Sort speakers by overlap
                sorted_overlaps = sorted(speaker_to_overlap.items(), key=lambda x: x[1], reverse=True)
                best_raw_id, best_overlap = sorted_overlaps[0]
                second_overlap = sorted_overlaps[1][1] if len(sorted_overlaps) > 1 else 0.0

                # Apply hysteresis with respect to previous speaker if available
                prev_overlap = speaker_to_overlap.get(prev_speaker - 1, 0.0) if prev_speaker else 0.0

                # Keep previous speaker if it is close to the best
                if prev_speaker and prev_overlap >= best_overlap * (1 - hysteresis_margin):
                    chosen = prev_speaker
                # Or if best is not clearly dominant over second best
                elif best_overlap < (second_overlap * (1 + hysteresis_margin)) and prev_speaker:
                    chosen = prev_speaker
                else:
                    chosen = best_raw_id + 1  # Convert to 1-based indexing
            else:
                # Fallback: keep previous speaker if any, else default to 1
                chosen = prev_speaker or 1

            # Assign speaker to all tokens in this utterance
            for i in range(s_idx, e_idx + 1):
                tokens[i].speaker = chosen
                logger.info(f"Assigned utterance token '{tokens[i].text}' [{tokens[i].start:.2f}-{tokens[i].end:.2f}] to consistent Speaker {chosen}")
            
            prev_speaker = chosen
    
    def _fix_short_segments(self, tokens):
        """
        Fix only very short speaker segments that are likely errors.
        Much more conservative than full smoothing.
        """
        if len(tokens) < 3:
            return
        
        # Only fix segments that are 1 token long and surrounded by same speaker
        i = 1
        while i < len(tokens) - 1:
            current_speaker = tokens[i].speaker
            prev_speaker = tokens[i-1].speaker
            next_speaker = tokens[i+1].speaker
            
            # If this is a single token surrounded by the same speaker, merge it
            if (prev_speaker == next_speaker and 
                current_speaker != prev_speaker and
                tokens[i].end - tokens[i].start < 0.2):  # Very short token
                
                tokens[i].speaker = prev_speaker
                logger.info(f"Fixed short segment: '{tokens[i].text}' changed from Speaker {current_speaker} to Speaker {prev_speaker}")
            
            i += 1
    
    def _smooth_speaker_assignments(self, tokens):
        """
        Smooth speaker assignments to prevent word splitting and improve coherence.
        Groups consecutive tokens that should belong to the same speaker.
        """
        if len(tokens) < 2:
            return
        
        # Parameters for smoothing - made more conservative
        MIN_SPEAKER_DURATION = 0.2  # Minimum duration for a speaker segment (reduced)
        MAX_GAP_WITHIN_SPEAKER = 0.1  # Maximum gap to consider same speaker (reduced)
        
        # Group tokens into potential speaker segments
        current_speaker = tokens[0].speaker
        current_start_idx = 0
        
        i = 1
        while i < len(tokens):
            # Check if we should continue with current speaker or switch
            should_switch = False
            
            # If speaker changes, check if it's a real change or noise
            if tokens[i].speaker != current_speaker:
                # Calculate duration of current speaker segment
                current_duration = tokens[i-1].end - tokens[current_start_idx].start
                
                # If current segment is very short, extend it
                if current_duration < MIN_SPEAKER_DURATION:
                    tokens[i].speaker = current_speaker
                else:
                    # Check gap between previous token and current token
                    gap = tokens[i].start - tokens[i-1].end
                    
                    # If gap is small, might be same speaker
                    if gap < MAX_GAP_WITHIN_SPEAKER:
                        # Look ahead to see if next few tokens have same speaker
                        lookahead_speaker = tokens[i].speaker
                        lookahead_count = 0
                        j = i
                        while j < len(tokens) and j < i + 3:  # Look ahead 3 tokens
                            if tokens[j].speaker == lookahead_speaker:
                                lookahead_count += 1
                            j += 1
                        
                        # If lookahead shows consistent speaker, switch
                        if lookahead_count >= 2:
                            should_switch = True
                        else:
                            # Keep current speaker
                            tokens[i].speaker = current_speaker
                    else:
                        should_switch = True
            
            if should_switch:
                current_speaker = tokens[i].speaker
                current_start_idx = i
            
            i += 1
        
        # Final pass: merge very short speaker segments
        self._merge_short_segments(tokens)
    
    def _merge_short_segments(self, tokens):
        """
        Merge speaker segments that are too short to be meaningful.
        """
        if len(tokens) < 3:
            return
        
        MIN_SEGMENT_TOKENS = 2  # Minimum tokens per speaker segment
        
        i = 0
        while i < len(tokens):
            current_speaker = tokens[i].speaker
            segment_start = i
            
            # Find end of current speaker segment
            while i < len(tokens) and tokens[i].speaker == current_speaker:
                i += 1
            
            segment_length = i - segment_start
            
            # If segment is too short, merge with adjacent segment
            if segment_length < MIN_SEGMENT_TOKENS:
                # Decide which adjacent segment to merge with
                merge_with_previous = False
                merge_with_next = False
                
                if segment_start > 0:  # Has previous segment
                    merge_with_previous = True
                elif i < len(tokens):  # Has next segment
                    merge_with_next = True
                
                if merge_with_previous:
                    # Merge with previous segment
                    target_speaker = tokens[segment_start - 1].speaker
                    for j in range(segment_start, i):
                        tokens[j].speaker = target_speaker
                elif merge_with_next:
                    # Merge with next segment
                    target_speaker = tokens[i].speaker if i < len(tokens) else current_speaker
                    for j in range(segment_start, i):
                        tokens[j].speaker = target_speaker
        

    
    def find_consistent_speaker_id(self, embedding: np.ndarray, original_speaker_id: str) -> int:
        """
        Find the consistent speaker ID for a given embedding using clustering with dynamic detection.
        """
        # Track this speaker as seen
        self.known_speakers.add(original_speaker_id)
        
        if embedding is None:
            # Fallback without embedding - check if we've seen this speaker before
            if original_speaker_id in self.speaker_id_map:
                return self.speaker_id_map[original_speaker_id]
            else:
                # New speaker without embedding - assign new ID
                new_speaker_id = self.next_speaker_id
                self.next_speaker_id += 1
                self.speaker_id_map[original_speaker_id] = new_speaker_id
                logger.info(f"Created new Speaker {new_speaker_id} for {original_speaker_id} (no embedding)")
                return new_speaker_id
        
        # If we already have a mapping for this original speaker, use it
        if original_speaker_id in self.speaker_id_map:
            consistent_id = self.speaker_id_map[original_speaker_id]
            # Add this embedding to the speaker's profile
            self.speaker_embeddings[consistent_id].append(embedding)
            return consistent_id
        
        # Find the best matching existing speaker using cosine similarity
        best_match_id = None
        best_similarity = 0.0
        similarity_threshold = 0.70  # Slightly lower threshold for better detection
        
        for speaker_id, embeddings in self.speaker_embeddings.items():
            if embeddings:
                # Calculate average embedding for this speaker
                avg_embedding = np.mean(embeddings, axis=0)
                similarity = cosine_similarity([embedding], [avg_embedding])[0][0]
                
                if similarity > best_similarity and similarity > similarity_threshold:
                    best_similarity = similarity
                    best_match_id = speaker_id
        
        if best_match_id is not None:
            # Found a matching speaker
            self.speaker_id_map[original_speaker_id] = best_match_id
            self.speaker_embeddings[best_match_id].append(embedding)
            logger.info(f"Mapped {original_speaker_id} to existing Speaker {best_match_id} (similarity: {best_similarity:.3f})")
            return best_match_id
        else:
            # Create a new speaker ID - this handles late-joining speakers
            new_speaker_id = self.next_speaker_id
            self.next_speaker_id += 1
            self.speaker_id_map[original_speaker_id] = new_speaker_id
            self.speaker_embeddings[new_speaker_id].append(embedding)
            
            # Check if this is a late-joining speaker
            if len(self.known_speakers) > 2:
                logger.info(f"🎯 LATE-JOINING SPEAKER DETECTED: Created Speaker {new_speaker_id} for {original_speaker_id}")
            else:
                logger.info(f"Created new Speaker {new_speaker_id} for {original_speaker_id}")
            
            return new_speaker_id
    
    def update_audio_buffer(self, audio_chunk: np.ndarray):
        """
        Update the audio buffer with new audio data and track timing.
        """
        if audio_chunk is not None:
            # If buffer is empty, set start time
            if len(self.audio_buffer) == 0:
                import time
                self.audio_buffer_start_time = time.time()
            
            # Add new audio to buffer
            self.audio_buffer.extend(audio_chunk.flatten())
            
            # Update start time if buffer was rotated (maxlen exceeded)
            if len(self.audio_buffer) == self.audio_buffer.maxlen:
                # Buffer is full, calculate new start time
                buffer_duration = len(self.audio_buffer) / self.sample_rate
                current_time = time.time()
                self.audio_buffer_start_time = current_time - buffer_duration
    
    def get_audio_segment(self, start_time: float, end_time: float) -> np.ndarray:
        """
        Extract audio segment from buffer based on timing.
        """
        try:
            if len(self.audio_buffer) == 0:
                logger.warning(f"Audio buffer is empty for segment [{start_time:.2f}-{end_time:.2f}]")
                return None
            
            # Calculate current time and buffer duration
            import time
            current_time = time.time()
            buffer_duration = len(self.audio_buffer) / self.sample_rate
            buffer_end_time = current_time
            buffer_start_time = buffer_end_time - buffer_duration
            
            # Check if requested segment is within buffer timeframe
            if end_time < buffer_start_time or start_time > buffer_end_time:
                logger.warning(f"Audio segment [{start_time:.2f}-{end_time:.2f}] outside buffer range [{buffer_start_time:.2f}-{buffer_end_time:.2f}]")
                return None
            
            # Calculate relative positions in buffer
            relative_start = max(0, start_time - buffer_start_time)
            relative_end = min(buffer_duration, end_time - buffer_start_time)
            
            # Convert to sample indices
            start_sample = int(relative_start * self.sample_rate)
            end_sample = int(relative_end * self.sample_rate)
            
            # Convert buffer to numpy array and extract segment
            audio_array = np.array(list(self.audio_buffer))
            
            if start_sample >= 0 and end_sample <= len(audio_array) and start_sample < end_sample:
                segment = audio_array[start_sample:end_sample]
                logger.info(f"Successfully extracted audio segment [{start_time:.2f}-{end_time:.2f}] from buffer")
                return segment
            else:
                logger.warning(f"Invalid sample range [{start_sample}-{end_sample}] for buffer size {len(audio_array)}")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to extract audio segment: {e}")
            return None

def concatenate_speakers(segments):
    segments_concatenated = [{"speaker": 1, "begin": 0.0, "end": 0.0}]
    for segment in segments:
        speaker = extract_number(segment.speaker) + 1
        if segments_concatenated[-1]['speaker'] != speaker:
            segments_concatenated.append({"speaker": speaker, "begin": segment.start, "end": segment.end})
        else:
            segments_concatenated[-1]['end'] = segment.end
    # print("Segments concatenated:")
    # for entry in segments_concatenated:
    #     print(f"Speaker {entry['speaker']}: {entry['begin']:.2f}s - {entry['end']:.2f}s")   
    return segments_concatenated


def add_speaker_to_tokens(segments, tokens):
    """
    Assign speakers to tokens based on diarization segments, with punctuation-aware boundary adjustment.
    """
    punctuation_marks = {'.', '!', '?'}
    punctuation_tokens = [token for token in tokens if token.text.strip() in punctuation_marks]
    segments_concatenated = concatenate_speakers(segments)
    for ind, segment in enumerate(segments_concatenated):
            for i, punctuation_token in enumerate(punctuation_tokens):
                if punctuation_token.start > segment['end']:
                    after_length = punctuation_token.start - segment['end']
                    before_length = segment['end'] - punctuation_tokens[i - 1].end
                    if before_length > after_length:
                        segment['end'] = punctuation_token.start
                        if i < len(punctuation_tokens) - 1 and ind + 1 < len(segments_concatenated):
                            segments_concatenated[ind + 1]['begin'] = punctuation_token.start
                    else:
                        segment['end'] = punctuation_tokens[i - 1].end
                        if i < len(punctuation_tokens) - 1 and ind - 1 >= 0:
                            segments_concatenated[ind - 1]['begin'] = punctuation_tokens[i - 1].end
                    break

    last_end = 0.0
    for token in tokens:
        start = max(last_end + 0.01, token.start)
        token.start = start
        token.end = max(start, token.end)
        last_end = token.end

    ind_last_speaker = 0
    for segment in segments_concatenated:
        for i, token in enumerate(tokens[ind_last_speaker:]):
            if token.end <= segment['end']:
                token.speaker = segment['speaker']
                ind_last_speaker = i + 1
                # print(
                #     f"Token '{token.text}' ('begin': {token.start:.2f}, 'end': {token.end:.2f}) "
                #     f"assigned to Speaker {segment['speaker']} ('segment': {segment['begin']:.2f}-{segment['end']:.2f})"
                # )
            elif token.start > segment['end']:
                break
    return tokens


def visualize_tokens(tokens):
    conversation = [{"speaker": -1, "text": ""}]
    for token in tokens:
        speaker = conversation[-1]['speaker']
        if token.speaker != speaker:
            conversation.append({"speaker": token.speaker, "text": token.text})
        else:
            conversation[-1]['text'] += token.text
    print("Conversation:")
    for entry in conversation:
        print(f"Speaker {entry['speaker']}: {entry['text']}")