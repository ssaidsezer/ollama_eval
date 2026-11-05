
import os
from numpy import rint
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Config
AUDIO_PATH = "ATBK_Pipeline/data/audio/benchmark/02"
MODEL_NAME = "openai/whisper-small"

def test_transcribe(device):
    print(f"\nTesting on {device}...")
    try:
        # Check if we should force float32 for MPS
        if device == "mps":
            dtype = torch.float32
        else:
            dtype = torch.float16 if device != "cpu" else torch.float32
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        model.to(device)
        
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=1,
            torch_dtype=dtype,
            device=device,
        )

        question_files = []

        print(AUDIO_PATH)
        for file in os.listdir(AUDIO_PATH):
            if file.endswith(".wav"):
                test_file = {
                    "id": file.split(".")[0],
                    "path": os.path.join(AUDIO_PATH, file),
                    "text": ""
                }
                question_files.append(test_file)
            
                
        for test_file in question_files:
            print(test_file['path'])

            result = pipe(test_file['path'], generate_kwargs={"language": "turkish"})
            print(f"Result ({device}): '{result['text']}'")
        
        
        
    except Exception as e:
        print(f"Error on {device}: {e}")

if __name__ == "__main__":
    if os.path.exists(AUDIO_PATH):
        # Test CPU first (Baseline)
        test_transcribe("cpu")
        
        # Test MPS if available
        if torch.backends.mps.is_available():
            test_transcribe("mps")
        else:
            print("MPS not available.")
    else:
        print(f"Audio file not found: {AUDIO_PATH}")
