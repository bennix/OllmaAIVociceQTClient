import sys
import pyaudio
import wave
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QComboBox, QLabel, QTextEdit, QMessageBox
import whisper
import ollama

# 录音设置
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
WAVE_OUTPUT_FILENAME = "output.wav"

class AudioRecorder(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.p = pyaudio.PyAudio()  # 确保在界面初始化之前初始化 pyaudio 对象
        self.initUI()
        self.is_recording = False
        self.stream = None
        self.frames = []
        self.markdown_content = "# Ollma AI Voice-assiant\n\n"
        self.update_markdown()
        self.whisper_model = whisper.load_model('medium')  
        self.transcript = ""

    def initUI(self):
        self.setWindowTitle('AI Voice Assiant')
        
        self.layout = QHBoxLayout()
        
        # 左侧布局 - 录音控制
        control_layout = QVBoxLayout()
        
        self.device_label = QLabel("Select an input device:")
        control_layout.addWidget(self.device_label)

        self.device_combo = QComboBox()
        control_layout.addWidget(self.device_combo)
        self.list_devices()
        
        self.record_button = QPushButton("Start AI Talking")
        self.record_button.clicked.connect(self.toggle_recording)
        control_layout.addWidget(self.record_button)

        self.status_label = QLabel("Press 'Start Talk' to begin.")
        control_layout.addWidget(self.status_label)

        control_container = QWidget()
        control_container.setLayout(control_layout)
        self.layout.addWidget(control_container)

        # 右侧布局 - Markdown 显示
        self.markdown_display = QTextEdit()
        self.markdown_display.setReadOnly(True)
        self.layout.addWidget(self.markdown_display)

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

    def list_devices(self):
        device_count = self.p.get_device_count()
        for i in range(device_count):
            device_info = self.p.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                self.device_combo.addItem(device_info['name'], i)

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
            self.add_markdown_content("YOU:")
        else:
            self.stop_recording()
            self.add_markdown_content("YOU[DAID]")

    def start_recording(self):
        self.is_recording = True
        self.record_button.setText("Stop Talk")
        self.status_label.setText("Talking... Press 'Stop Talk' to send a new query.")
        self.frames = []
        device_index = self.device_combo.currentData()

        try:
            self.stream = self.p.open(format=FORMAT,
                                      channels=CHANNELS,
                                      rate=RATE,
                                      input=True,
                                      input_device_index=device_index,
                                      frames_per_buffer=CHUNK,
                                      stream_callback=self.callback)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.is_recording = False
            self.record_button.setText("Start Talkiing")
            self.status_label.setText("Press 'Start Talk' to begin.")
            return

    def stop_recording(self):
        self.is_recording = False
        self.stream.stop_stream()
        self.stream.close()
        
        self.record_button.setText("Start Recording")
        self.status_label.setText(f"Recording stopped. File saved as {WAVE_OUTPUT_FILENAME}")

        # 写入 WAV 文件
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        self.transcript = self.whisper_model.transcribe(WAVE_OUTPUT_FILENAME)['text']
        self.add_markdown_content(self.transcript)
        self.response = ollama.chat(model='gemma2', messages=[
                                {
                                    'role': 'user',
                                    'content': self.transcript,
                                },
                                ])
        self.add_markdown_content(self.response['message']['content'])


    def callback(self, in_data, frame_count, time_info, status):
        if status:
            print(f"Error: {status}")
        self.frames.append(in_data)
        if not self.is_recording:
            return (None, pyaudio.paComplete)
        return (None, pyaudio.paContinue)

    def closeEvent(self, event):
        self.p.terminate()  # 确保在应用程序关闭时终止 pyaudio 对象

    def add_markdown_content(self, content):
        self.markdown_content += f"\n{content}\n"
        self.update_markdown()

    def update_markdown(self):
        self.markdown_display.setMarkdown(self.markdown_content)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    recorder = AudioRecorder()
    recorder.show()
    sys.exit(app.exec_())
