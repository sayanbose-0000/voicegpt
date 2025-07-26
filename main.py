from ctranslate2.models import WhisperGenerationResultAsync
from llama_cpp import Llama, time
import os
import pyaudio
import wave
import keyboard
from faster_whisper import WhisperModel
import edge_tts
import asyncio
import pygame

async def speak(text):
  file = "recordings/outputaudio.mp3"
  tts = edge_tts.Communicate(text, voice="en-US-AndrewMultilingualNeural")
  await tts.save(file)
  
  pygame.mixer.init()
  pygame.mixer.music.load(file)
  pygame.mixer.music.play()
  
  while pygame.mixer.music.get_busy():
    pygame.time.wait(100)
    
  pygame.mixer.music.stop()
  pygame.mixer.quit()


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
  
  while(keyboard.is_pressed('down')):
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
  
def whisper_transcribe(whisper_model):
  
  segments, info = whisper_model.transcribe("recordings/myaudio.wav", beam_size=5)
  
  input_text=""
  for segment in segments:
    input_text += segment.text
    
  return input_text
  
def llama_model(input_text, message, llm):
  print("User: ", input_text)
  
  message.append({
    "role": "user",
    "content": input_text
  })
  
  response = llm.create_chat_completion(messages=message, stream=True)
  
  print("Assistant", end=": ")
  
  full_reply=""
  buffer=""
  count = 0
  for i in response:
    content = i["choices"][0]["delta"].get("content", "")
    print(content, end="", flush=True)
    count += 1
    full_reply += content
    
  print()
  message.append({
    "role": "assistant",
    "content": full_reply
  })
  
  # Text to speech
  # pyttsx3.speak(full_reply)
  asyncio.run(speak(full_reply))

def main():
  # Llama
  llm = Llama(
    model_path="models/Llama-3.2-1B-Instruct-IQ3_M.gguf",
    chat_format="llama-3",
    verbose=False,
    n_ctx=2048
  )

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
  
  # Whisper
  whisper_model_path = "models/faster_whisper_tiny.en_model"
  whisper_model = WhisperModel(whisper_model_path, device="cpu", compute_type="int8")

  
  os.system("clear")
  
  print("Press and hold down arrow to speak")
  while (True):
    
    while (keyboard.is_pressed('down')):
      record_voice() 
      
      input_text = whisper_transcribe(whisper_model)
      # os.system("clear")
      
      llama_model(input_text, message, llm)
      print()
      print("Press and hold down arrow to speak")
      
    continue

main()
