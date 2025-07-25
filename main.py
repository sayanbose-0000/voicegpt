from ctranslate2.models import WhisperGenerationResultAsync
from llama_cpp import Llama
import os
from numpy._core.numeric import result_type
import pyaudio
import wave
import keyboard
from faster_whisper import WhisperModel

def record_voice():
  CHUNK = 1024 
  FORMAT = pyaudio.paInt16
  CHANNELS = 1
  RATE = 16000
  RECORD_SECONDS = 5
  WAVE_OUTPUT_FILENAME = "recordings/myaudio.wav"
  
  p = pyaudio.PyAudio()
  stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=8192
  )
  
  frames = []
  
  # os.system("clear")
  print("Hold space to Start speaking...")
  
  # while (True):
  #   if (keyboard.is_pressed('q')):
  #     print("Recording stopped")
  #     break
  
  while(keyboard.is_pressed('space')):
    data = stream.read(CHUNK)
    frames.append(data)
  
  stream.stop_stream()
  stream.close()
  p.terminate()
  
  # Save the recorded audio file
  wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
  wf.setnchannels(CHANNELS)
  wf.setsampwidth(p.get_sample_size(FORMAT))
  wf.setframerate(RATE)
  wf.writeframes(b''.join(frames))
  wf.close()
  
def whisper_transcribe():
  model_path = "models/faster_whisper_tiny.en_model"
  model = WhisperModel(model_path, device="cpu", compute_type="int8")
  
  segments, info = model.transcribe("recordings/myaudio.wav", beam_size=5)
  
  input_text=""
  for segment in segments:
    input_text += segment.text
    
  return input_text
  
def llama_model(input_text):
  llm = Llama(
    model_path="models/Llama-3.2-1B-Instruct-Q8_0.gguf",
    chat_format="llama-3",
    verbose=False,
    n_ctx=8192
  )

  print("User: ", input_text)

  message=[
    {
      "role": "system",
      "content": (
        "You are a concise, knowledgeable assistant. Keep answers short, precise, and on-topic. "
        "Prioritize factual accuracy, and search when needed (assume internet access). "
        "Avoid unnecessary styling or emojis. Rarely use bullet points, and never nest them. "
        "Maintain a professional tone, use plain language, and subtle humor if appropriate. "
        "You are currently chatting with a developer or technically inclined user."
      )
    }
  ]
  
  message.append({
    "role": "user",
    "content": input_text
  })
  
  response = llm.create_chat_completion(messages=message, stream=True)
  
  print("Assistant", end=": ")
  
  full_reply=""
  for i in response:
    content = i["choices"][0]["delta"].get("content", "")
    print(content, end="", flush=True)
    full_reply += content
    
  print()
  message.append({
    "role": "assistant",
    "content": full_reply
  })

def main():
  while (True):
    record_voice()
    # os.system("clear")
    
    input_text = whisper_transcribe()
    # os.system("clear")
    
    
    llama_model(input_text)

main()
