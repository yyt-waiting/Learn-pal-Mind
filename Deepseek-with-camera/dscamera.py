import os
import cv2
import time
import io
import threading
import queue
import numpy as np
import pyaudio
import wave
import keyboard
import customtkinter as ctk
from PIL import Image, ImageTk
import oss2
from pydub import AudioSegment
from pydub.playback import play
from openai import OpenAI
import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from datetime import datetime
import re
import logging

# ---------------- Configuration ----------------


# TTS Configuration
dashscope.api_key = QWEN_API_KEY
TTS_MODEL = "cosyvoice-v1"
TTS_VOICE = "longwan"

# SenseVoice ASR Configuration
MODEL_DIR = "iic/SenseVoiceSmall"

# Audio Recording Configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
WAVE_OUTPUT_FILENAME = "output.wav"

# Logging Configuration
LOG_FILE = "behavior_log.txt"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ---------------- API Clients Initialization ----------------
# DeepSeek Client 开始创建所需要调用的API客户端对象
deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

# Qwen-VL Client
qwen_client = OpenAI(
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL
)

# ASR Model
asr_model = AutoModel(
    model=MODEL_DIR,
    trust_remote_code=True,
    remote_code="./model.py",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
)


#配置说明：OSS 配置：
# 是一款高可靠、安全、低成本、高扩展性的分布式对象存储服务。 它可以帮助用户轻松地存储和管理海量非结构化数据
# 用处：_upload_screenshots() 会用 oss2.Auth + oss2.Bucket 把图片上传到阿里云 OSS，
# 然后生成可公开访问的 URL 传给 Qwen-VL。


# ---------------- Utility Functions ----------------
def extract_language_emotion_content(text):
    """Extract clean content from ASR output"""
    # Extract language
    language_start = text.find("|") + 1
    language_end = text.find("|", language_start)
    language = text[language_start:language_end]
    
    # Extract emotion
    emotion_start = text.find("|", language_end + 1) + 1
    emotion_end = text.find("|", emotion_start)
    emotion = text[emotion_start:emotion_end]
    
    # Extract content
    content_start = text.find(">", emotion_end) + 1
    content = text[content_start:]
    
    # Clean up any remaining tags
    while content.startswith("<|"):
        end_tag = content.find(">", 2) + 1
        content = content[end_tag:]
    
    return content.strip()

def extract_behavior_type(analysis_text):
    """Extract behavior type number from AI analysis text"""
    # Try to find behavior type number in the text (1-7)
    pattern = r'(\d+)\s*[.、:]?\s*(认真专注工作|吃东西|用杯子喝水|喝饮料|玩手机|睡觉|其他)'
    match = re.search(pattern, analysis_text)
    
    if match:
        behavior_num = match.group(1)
        behavior_desc = match.group(2)
        return behavior_num, behavior_desc
    
    # Alternative pattern if the first one fails
    patterns = [
        (r'认真专注工作', '1'),
        (r'吃东西', '2'),
        (r'用杯子喝水', '3'),
        (r'喝饮料', '4'),
        (r'玩手机', '5'),
        (r'睡觉', '6'),
        (r'其他', '7')
    ]
    
    for pattern, num in patterns:
        if re.search(pattern, analysis_text):
            return num, pattern
    
    return "0", "未识别"  # Default if no pattern matches

# ---------------- Camera Display Window ----------------
class CameraWindow(ctk.CTkToplevel):
    #两个文件的 CameraWindow 虽然名字相同且都是继承自 CTkToplevel，但针对的功能和上下文不同。
    
    #和diagram文件不同->复习！
    #     diagram文件：类名：WebcamHandler
    # 功能：负责摄像头采集、帧处理、图像分析触发及截图上传
    # 核心方法：
    # start：启动摄像头采集线程，初始化摄像头窗口
    # stop：停止摄像头采集，释放资源
    # _process_webcam：持续读取摄像头帧，更新最新画面
    # capture_and_analyze：触发图像捕获与分析流程
    # _capture_screenshots：连续捕获多帧画面用于分析
    # _upload_screenshots：将截图上传至 OSS 存储
    # _get_image_analysis：调用 Qwen-VL API 分析图像内容

    #而在dscamera.py文件中（本文件：只有update_frame() on_closing()两个函数）

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Camera Feed")
        self.geometry("640x480")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Create frame for the camera display
        self.camera_frame = ctk.CTkFrame(self)
        self.camera_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create label for the camera image
        self.camera_label = ctk.CTkLabel(self.camera_frame, text="Starting camera...")
        self.camera_label.pack(fill="both", expand=True)
        
        # Image holder
        self.current_image = None
        
        # Flag to indicate if window is closed
        self.is_closed = False
    
    def update_frame(self, img):
        """Update camera frame with new image"""
        if self.is_closed:
            return
            
        try:
            if img:
                # Resize the image to fit the window nicely
                img_resized = img.copy()
                img_resized.thumbnail((640, 480))
                
                # Convert to CTkImage
                ctk_img = ctk.CTkImage(light_image=img_resized, dark_image=img_resized, size=(640, 480))
                
                # Update the label
                self.camera_label.configure(image=ctk_img, text="")
                
                # Store a reference to prevent garbage collection
                self.current_image = ctk_img
        except Exception as e:
            print(f"Error updating camera frame: {e}")
    
    def on_closing(self):
        """Handle window close event"""
        self.is_closed = True
        self.withdraw()  # Hide instead of destroy to allow reopening

# ---------------- Core Functionality Classes ----------------
class AudioRecorder:
    #这个类实现了“按键开始录音，按键停止录音”的功能，
    #通过后台线程实时从麦克风采集音频数据，录完后保存为 WAV 文件，并调用主程序转写。
    def __init__(self, app):
        # 这个app再次出现，再次说明：

        # MultimediaAssistantApp() 是程序的主窗口类->代码最下面
        # 继承自 ctk.CTk，这是 customtkinter 框架里自定义的主窗口类，它是整个应用的顶层窗口和主程序核心控制器
        # 你仔细看对应的代码会看到下面：
        # self.audio_recorder = AudioRecorder(self)
        # self.webcam_handler = WebcamHandler(self)
        # self.audio_player = AudioPlayer(self)
        # self.voice_detector = VoiceActivityDetector(self)
        # 这里把自己的实例 self 传给了这四个子模块说明它是所有模块的“协调者”和“数据中心”
        #子模块们通过 self.app（指向 MultimediaAssistantApp 的实例）调用主程序的功能，比如界面更新、转写处理、日志记录、消息队列等
        #MultimediaAssistantApp 持有这四个模块实例，可以随时调用它们的方法->self.app.xxxx随便调用！！！
        self.app = app
        self.recording = False
        self.stop_recording_flag = False
        #recording 和 stop_recording_flag 是状态标志，控制录音线程的启动与停止
        self.audio_thread = None
        #audio_thread 保存后台录音线程的引用
        
    def start_recording(self):
        """Begin audio recording when 'r' key is pressed"""
        if not self.recording:
            self.recording = True
            #start_recording：启动录音逻辑
            self.stop_recording_flag = False#同上，关闭逻辑
            self.audio_thread = threading.Thread(target=self._record_audio)
            #创建后台线程执行 _record_audio 方法，设置为守护线程（程序关闭时自动退出）
            self.audio_thread.daemon = True
            self.audio_thread.start()
            self.app.update_status("Recording...")
            #更新 UI 状态显示“Recording...”
    
    def stop_recording(self):
        """Stop audio recording when 's' key is pressed"""
        if self.recording:
            self.stop_recording_flag = True
            #停止录音逻辑
            self.recording = False
            if self.audio_thread and self.audio_thread.is_alive():
                self.audio_thread.join(timeout=1.0)
                #如果录音线程仍存活，调用 join() 等待线程最多1秒安全退出
                #self.audio_thread.is_alive()：来自你开头import 的 threading库函数！
                #.is_alive() 是 Python 线程对象的方法，返回布尔值：
                #等待线程最多1秒让它结束，确保资源被正确释放，不会造成程序异常或僵死。
            self.app.update_status("Processing audio...")
            #切换 UI 状态到“Processing audio...”提示用户录音结束，正在处理
    
    def _record_audio(self):
        #_record_audio 是后台线程执行的方法，真正从麦克风采集数据
        """Record audio from microphone"""
        p = pyaudio.PyAudio()
        #pyaudio.PyAudio() 创建 PyAudio 对象
        #PyAudio 是一个 Python 的音频接口库，用于访问麦克风和音响设备。
        #通过这个对象，你可以访问音频设备并执行音频操作，例如录制、播放和实时处理音频。
        stream = p.open(
                      format=FORMAT,#录音数据的格式，通常是 16-bit 整数 PCM 格式
                      channels=CHANNELS,#（单声道）。声道数，1 表示单声道，2 是立体声。语音录音一般用单声道。
                      rate=RATE,#（采样率，16kHz）
                      input=True,#表示录音流，表示这是输入流（录音流），False 表示输出流（播放流）。
                      frames_per_buffer=CHUNK#缓冲区大小，一次从设备读取多少帧数据。
                      )
        #通过这个 stream，你可以调用 stream.read(CHUNK) 读取音频数据。
        frames = []
        #用来存放录制的音频数据块（字节串）。
        
        while self.recording and not self.stop_recording_flag:
            #这是循环录音的条件，只要没被停止，就持续录音。
            try:
                data = stream.read(CHUNK)
                #从麦克风一次性读 CHUNK 大小的音频数据（字节串）。这个操作会阻塞，直到读到足够数据。
                frames.append(data)
                #将这次读取的音频数据保存起来，后面用来写文件。
            except Exception as e:
                self.app.update_status(f"Error recording audio: {e}")
                break
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        # 录音结束后：
        # 停止流 stream.stop_stream()
        # 关闭流 stream.close()
        # 释放 PyAudio 资源 p.terminate()
        
        if frames:# 判断录到音频才写文件：
            try:
                wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
                #  wave 模块是一个标准库模块，用于处理 WAV 格式的音频文件。
                #  它支持对未压缩的单声道或立体声 WAV 文件进行读写操作。以下是模块的主要功能和用法：
                #wf 其实是 wave 模块 里打开的 WAV 文件对象，名字叫 wf，它代表的是一个“WAV 文件写入器”的实例。
                wf.setnchannels(CHANNELS)#设置 WAV 文件的声道数（单声道或立体声）
                wf.setsampwidth(p.get_sample_size(FORMAT))
                #设置采样宽度（每个采样点占用多少字节），这里用 p.get_sample_size(FORMAT) 自动获得对应的字节数，比如16位采样就是2字节
                wf.setframerate(RATE)#设置采样率，表示每秒采样多少次
                wf.writeframes(b''.join(frames))
                #把之前录到的所有音频数据块 frames 拼接成一个完整的字节流，写入 WAV 文件
                wf.close()
                
                self.app.transcribe_audio(WAVE_OUTPUT_FILENAME)
                #调用主程序（self.app）中的一个方法 transcribe_audio，
                #把刚录制并保存好的音频文件路径（WAVE_OUTPUT_FILENAME）传给它，让主程序去做“音频转文字”的处理。
            except Exception as e:
                self.app.update_status(f"Error saving audio: {e}")

class VoiceActivityDetector:
    def __init__(self, app):
        #初始化的信息炸弹！！！！！！！！
        self.app = app
        #又是你！经典的获取类的权限的钥匙！保存主程序实例，方便后续调用
        self.running = False
        #定义标志变量，表示语音检测线程是否在运行
        self.listening_thread = None
        #初始化监听线程变量，后面启动监测时会赋值为线程对象
        self.detection_thread = None
        #虽然这个类中未使用，但通常用于识别或处理线程的占位变量，准备扩展用
        
        # Voice activity detection parameters - MUCH lower threshold
        #初始化语音检测参数
        self.energy_threshold = 80  # Further reduced for better sensitivity
        #语音信号能量判定阈值，能量高于它才算是有人说话
        self.dynamic_threshold = True  # Dynamically adjust threshold based on environment noise
        #是否根据环境噪声自动调整阈值，提升适应性
        self.silence_threshold = 0.8  # Seconds of silence to consider speech ended
        #语音结束判定的静音时长，超过则认为说话结束
        self.min_speech_duration = 0.3  # Shorter minimum duration to catch brief utterances
        #最短语音长度，避免误触发
        self.max_speech_duration = 30.0  # Maximum speech duration
        #最长语音长度，防止录音过长
        
        # Speech detection state
        # 初始化语音检测状态变量
        self.is_speaking = False
        #当前是否处于“说话中”的状态
        self.speech_started = 0
        #语音开始时间戳
        self.silence_started = 0
        #静音开始时间戳
        self.speech_frames = []
        #保存检测到的语音音频帧，供后续识别处理用
        
        # For dynamic threshold adjustment
        #初始化动态阈值相关变量
        self.noise_levels = []
        #保存环境噪声的历史能量值，用于计算平均噪声
        self.max_noise_levels = 100
        #保存噪声样本的最大数量，防止内存无限增长
        
        # Audio stream
        self.audio = None
        self.stream = None
        #self.audio 和 self.stream 是后续打开音频采集设备和流的句柄，先初始化为空，稍后在监听线程中赋值
        
        # Debug mode调试和校准相关变量
        self.debug = True  # Set to True to enable energy level debugging
        #是否打印调试信息，方便你观察能量变化和检测过程
        
        # Add a calibration phase
        self.is_calibrating = True
        #表示是否处于麦克风噪声校准阶段
        self.calibration_duration = 3  # seconds
        #校准持续时间，默认3秒
        self.calibration_start_time = 0
        #记录校准开始时间戳，用于控制校准时长
    
    def start_monitoring(self):
        """Begin continuous voice monitoring"""
        if not self.running:
            self.running = True

            self.listening_thread = threading.Thread(target=self._monitor_audio)
            self.listening_thread.daemon = True#守护进程
            #守护进程，守护的是创建它的进程（下称“A进程”），如果A结束了，守护进程也就结束了。
            #启动一个后台线程 listening_thread，异步执行音频采集和语音检测任务
            self.listening_thread.start()
            self.app.update_status("语音监测启动中... 正在校准麦克风")
    
    def stop_monitoring(self):
        """Stop voice monitoring"""
        self.running = False
        if self.listening_thread and self.listening_thread.is_alive():
            #注意：self.listening_thread是一个线程对象，而不是一个bool值，意思是：如果 self.listening_thread 是 None → 结果是 False
            self.listening_thread.join(timeout=1.0)
            #使用 join() 等待线程优雅结束，最多等待 1 秒
        if self.audio and self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
            # properly close and clean up the PyAudio instance
            # terminate:结束；停止；终结
            # Qus：为什么self.stream and self.audio会有内置函数？他们不是上面弄刚设置None空对象吗？下面也是空对象呀
            #because：在_monitor_audio()方法中：self.audio = pyaudio.PyAudio()创建一个新的 PyAudio 对象（实例），
            # self.stream = self.audio.open()使得self.stream同样具有该对象属性
            self.audio = None
            self.stream = None
            # 可视化状态变化表
            # 阶段	self.audio	self.stream	硬件状态
            # 初始化	None	None	空闲
            # 开始录音	PyAudio实例	Stream实例	占用中
            # 录音中	PyAudio实例	Stream实例	占用中
            # 停止录音	None	None	空闲
            # 再次录音	新PyAudio实例	新Stream实例	占用中

            #关闭和释放 PyAudio 的流和实例，防止资源泄露或占用麦克风
    
    def _get_energy(self, audio_data):
        #计算传入音频数据帧的“能量”（声音强度）:能量是判断是否有人说话的关键特征
        """Calculate audio energy level"""
        try:
            # Convert bytes to numpy array
            data = np.frombuffer(audio_data, dtype=np.int16)
            #把原始 PCM 字节数据转成 16 位有符号整数的 numpy 数组，方便数值计算。
            #音频数据需要转变数据的格式
            
            # Ensure we have valid data
            if len(data) == 0 or np.all(data == 0):
                #如果没有采样点或全是静音，能量直接算 0。
                return 0.0
                
            # Calculate RMS energy
            # Use np.mean(np.abs(data)) as it's more robust than squaring
            energy = np.mean(np.abs(data))
            #计算该音频帧的平均振幅（绝对值均值），代表声音强度，用来判断有没有声音。
            return energy
        except Exception as e:
            print(f"Error calculating energy: {e}")
            return 0.0
    
    def _is_speech(self, audio_data, energy=None):
        #用来判断一段音频数据（audio_data）里是否包含语音，而不是纯背景噪音。
        #判断依据主要是 音频能量（energy）和 能量阈值（threshold）的比较。
        """Detect if audio chunk contains speech based on energy level"""
        try:
            # Skip speech detection if audio is playing
            if hasattr(self.app, 'is_playing_audio') and self.app.is_playing_audio:
                #hasattr 是 Python 内置的一个函数，用于判断一个对象是否拥有指定的属性或方法
                #hasattr(object, attribute)，object 为待检查的对象，attribute为属性或方法名
                #程序正在播放语音时，暂停检测，避免自说自话被误判。

                #debug输出，每2秒打印一次
                if self.debug and time.time() % 2 < 0.1:
                    print("语音监测暂停中 - 正在播放系统语音")
                return False
            
            # Use provided energy or calculate it
            if energy is None:
                energy = self._get_energy(audio_data)
                #如果调用者没传 energy 参数，就用 _get_energy(audio_data) 计算当前这段音频的能量值
                #音频能量 是声音强度的数值化表现，通常用来区分静音 / 说话
            # If we're calibrating（校准）, just collect noise levels


            #！！！！！！！！！！！非常重要！！！！！！！！！！！！
            if self.is_calibrating:#self.is_calibrating → 当前正在收集环境噪声的样本
                self.noise_levels.append(energy)
                return False
                #直接返回 False（不检测语音，因为这是纯噪声采集阶段）
                #非常重要：当_calibrate_microphone()正在运行时，_is_speech()不会检测语音，而是把能量值存进self.noise_levels供校准使用。

            
            # Adjust threshold dynamically if enabled
            threshold = self.energy_threshold
            #动态阈值调整:非常聪明！
            if self.dynamic_threshold and len(self.noise_levels) > 0:
                # Set threshold to be 2.5x the average noise level
                noise_avg = sum(self.noise_levels) / len(self.noise_levels)
                dynamic_threshold = noise_avg * 2.5
                threshold = max(threshold, dynamic_threshold)

                # threshold 初始是固定值 self.energy_threshold
                # 如果 self.dynamic_threshold 开启，并且已有噪声样本：
                # 计算环境噪声平均值 → noise_avg
                # 乘以系数 2.5 得到新的动态阈值
                # max(threshold, dynamic_threshold) 保证阈值不会低于初始固定值
                # 动态的阈值就是：动态阈值 = 环境噪声平均值 * 2.5，最终划定一条环境背景噪音的界限
                # 动态阈值可以适应不同环境，比如安静的办公室 vs 嘈杂的咖啡厅


            # Debug output for energy levels
            #每秒大约打印一次当前能量、阈值、平均噪音
            if self.debug and time.time() % 1 < 0.1:  # Print every second
                print(f"能量: {energy:.1f}, 阈值: {threshold:.1f}, " + 
                      f"平均噪音: {sum(self.noise_levels) / max(1, len(self.noise_levels)):.1f}")


            # Detect speech when energy is above threshold
            return energy > threshold
            #核心判定条件：当前音频的能量值是否高于阈值


        except Exception as e:
            print(f"Error in speech detection: {e}")
            return False
    
        #     运行流程总结
        # 如果自己在播放声音 → 不检测
        # 如果没提供能量值 → 先算一个
        # 如果是校准阶段 → 只记录噪音，不检测
        # 如果启用动态阈值 → 用环境噪声动态调整
        # 比较当前能量和阈值 → 决定是否为语音

        #     小测（升级版）
        # 假设：
        # self.energy_threshold = 100
        # self.dynamic_threshold = True
        # self.noise_levels = [40, 50, 60]
        # 当前音频能量 energy = 130
        # 问题：
        # 动态阈值会被设为多少？最终 threshold 是多少？最终会返回 True 还是 False？




    def _calibrate_microphone(self):
        # 麦克风环境噪音校准器。校准！
        #它的目标是：
        # 先采集一段时间的环境噪声样本（要求用户保持安静）。
        # 计算噪音平均能量值。
        # 把语音检测的能量阈值（self.energy_threshold）动态调整到 2.5 倍的平均噪音值，确保环境再吵也能正常识别。
        """Calibrate microphone by measuring background noise"""
        try:
            self.calibration_start_time = time.time()
            #记录校准开始时间，这样后面才能判断校准持续的时间是否达到了预设的 self.calibration_duration
            #就是说校准时间太慢或者太快都不准确！
            self.is_calibrating = True
            #开启校准状态
            self.noise_levels = []
            #清空旧的噪音样本列表，确保这次校准只用新的数据。
            print("开始麦克风校准...")
            self.app.update_status("校准麦克风中，请保持安静...")
            
            # Wait for calibration to complete
            while self.is_calibrating and time.time() - self.calibration_start_time < self.calibration_duration:
                time.sleep(0.1)
                #这里并没有直接采集噪音样本，因为采集是在 其他线程的 _is_speech 调用 中进行的（它会把环境能量追加到 self.noise_levels 里）。
                # 这段循环只是让程序停在这里，一直等到：
                # is_calibrating = False（提前结束），或者
                # 校准时间到达 self.calibration_duration（正常结束）。

            # Calculate noise threshold
            if len(self.noise_levels) > 0:
                #平均噪音能量值
                avg_noise = sum(self.noise_levels) / len(self.noise_levels)
                #设置语音检测阈值
                self.energy_threshold = max(100, avg_noise * 2.5)  # Set threshold to 2.5x average noise
                
                print(f"麦克风校准完成: 平均噪音级别 {avg_noise:.1f}, 阈值设为 {self.energy_threshold:.1f}")
                self.app.update_status(f"语音监测已启动 (阈值: {self.energy_threshold:.1f})")
            else:
                print("校准失败: 没有收集到噪音样本")
                self.app.update_status("语音监测已启动，但校准失败")
            
            self.is_calibrating = False

        #📌 兜底异常处理：如果中途出错，立刻停止校准，并给用户提示。
        except Exception as e:
            print(f"麦克风校准错误: {e}")
            self.is_calibrating = False
            self.app.update_status("语音监测已启动，但校准出错")
    
#上述两个函数的明显区别：

# _calibrate_microphone 主动发起一次噪音采集任务，让系统在短时间内收集环境噪音数据，用这些数据计算并设定一个合适的阈值。
# _is_speech 是实时语音检测器，它会不断处理麦克风音频片段，判断是否有人说话。
# 当 _calibrate_microphone 正在运行时，会设置 self.is_calibrating=True，这时 _is_speech 不会判断语音，而是把能量值存进 self.noise_levels 供校准使用。


# 持续从麦克风实时采集音频数据，检测和识别“有没有人在说话”，并在检测到语音开始、语音结束或语音过长时，
# 触发相应的处理逻辑（比如保存音频、调用后续语音识别等），同时管理音频流的打开和关闭。
    def _monitor_audio(self):
        """Continuously monitor audio for speech"""
        try:
            #看！上面的那个audio stream对象的问题本质解决了，问题就在这里！
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK#chunk：数据块，大块
            )
            #self.audio → 一个 PyAudio 的总控制对象，相当于“音频工厂”，用它来打开或关闭音频流。
            #self.stream → 一个正在录音的音频流对象，能从麦克风实时读取数据
            #format、channels、rate、frames_per_buffer 
            #→ 决定了录音质量、声道数、采样率、一次读取的音频块大小（CHUNK）。
            #参数设置相见最开始的代码“：前面有
                # # Audio Recording Configuration
                # CHUNK = 1024
                # FORMAT = pyaudio.paInt16
                # CHANNELS = 1
                # RATE = 16000
                # WAVE_OUTPUT_FILENAME = "output.wav"

            # Perform initial calibration
            self._calibrate_microphone()
            # 让程序先静静地听一小段时间背景音，收集噪音样本 self.noise_levels。
            # 最后计算 self.energy_threshold（能量阈值），用来区分“有说话”和“没说话”。
            # 校准结束后，self.is_calibrating = False，进入正式监控。
                        
            # Continuous audio analysis loop进入监听循环
            while self.running:
                try:
                    # Read audio chunk
                    audio_data = self.stream.read(CHUNK, exception_on_overflow=False)
                    #每一轮循环从麦克风抓一块音频数据（字节串）。大小是 CHUNK，对应大约几十毫秒的声音
                    #pyaudio 的Stream.read（） 有一个关键字参数exception_on_overflow，请将其设置为 False。避免输入溢出报错

                    # Calculate energy once to avoid duplicate work
                    energy = self._get_energy(audio_data)
                    
                    # Update noise level (only when not speaking)
                    if not self.is_speaking and len(self.noise_levels) < self.max_noise_levels:
                        self.noise_levels.append(energy)
                        #只有在当前没检测到说话时才更新 noise_levels（防止说话声音被当作噪音）。
                        #noise_levels 用来动态计算新的阈值（如果开启了 self.dynamic_threshold）

                        if len(self.noise_levels) > self.max_noise_levels:
                            self.noise_levels.pop(0)  # Keep the list size limited
                    
                    # Check if it's speech
                    #这是当前这一小段音频（一个 chunk，大约几毫秒）有没有检测到说话。
                    if self._is_speech(audio_data, energy):
                        # 判断是否说话

                        # If we weren't already speaking, mark the start
                        if not self.is_speaking:
                            #这是一个状态变量，表示系统之前是否已经进入“讲话状态”。
                            self.is_speaking = True
                            self.speech_started = time.time()
                            self.speech_frames = []
                            # Show visual feedback immediately
                            print("语音开始检测中...")
                            self.app.after(0, lambda: self.app.update_status("检测到语音输入..."))
                        
                        # Reset silence counter
                        self.silence_started = 0
                        
                        # Add frame to speech buffer
                        self.speech_frames.append(audio_data)
                        
                        # Check if we've exceeded max duration
                        if time.time() - self.speech_started > self.max_speech_duration:
                            print(f"达到最大语音长度 ({self.max_speech_duration}s)，开始处理")
                            self._process_speech()
                    
                    elif self.is_speaking:
                        # If we were speaking, but now detected silence
                        if self.silence_started == 0:
                            self.silence_started = time.time()
                            print(f"检测到语音之后的静音")
                        
                        # Add the silent frame (for smoother audio)
                        self.speech_frames.append(audio_data)
                        
                        # If silence continues for threshold duration, process the speech
                        silence_duration = time.time() - self.silence_started
                        if silence_duration > self.silence_threshold:
                            print(f"静音时长达到阈值 ({silence_duration:.2f}s > {self.silence_threshold}s)，开始处理语音")
                            self._process_speech()
                    
                    time.sleep(0.01)  # Small sleep to reduce CPU usage
                    
                except Exception as e:
                    error_msg = f"音频监测错误: {e}"
                    print(error_msg)
                    self.app.update_status(error_msg)
                    time.sleep(0.5)  # Sleep before retry

    # 难点总结：场景：

    # 原本安静：
    # _is_speech → False
    # self.is_speaking → False
    # → 什么也不做。

    # 突然有人开始说话：
    # _is_speech → True
    # self.is_speaking → False（之前还没说话）
    # → 进入讲话状态，记录时间，清空旧语音，提示 UI“检测到语音输入”。

    # 继续说话：
    # _is_speech → True
    # self.is_speaking → True（已经是讲话状态）
    # → 不再初始化，只是不断收集新音频到 speech_frames。

    # 说话超时：
    # 如果说了太久（time.time() - self.speech_started > max_speech_duration）
    # → 提前处理这段语音（防止无限长）。
                        


        except Exception as e:
            error_msg = f"语音监测失败: {e}"
            print(error_msg)
            self.app.update_status(error_msg)
        
        #清理资源：程序退出时停止并关闭音频流，释放麦克风设备。
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.audio:
                self.audio.terminate()
    



    def _process_speech(self):
        """Process detected speech segment"""
        #其实就是**当系统判断“讲话结束”**后，把那一整段录下来的音频交给后续处理（保存、转文字等）的步骤。
        speech_duration = time.time() - self.speech_started
        #此时的self.speech_started是在_monitor_audio(self)里面获得的时间戳time.time()
        
        # Only process if speech is long enough and has frames
        #要求：必须讲话时间够长（不小于 min_speech_duration 秒），必须录到了一些音频帧（speech_frames 列表非空）。
        if speech_duration >= self.min_speech_duration and len(self.speech_frames) > 0:
            print(f"处理语音片段: {speech_duration:.2f}秒, {len(self.speech_frames)} 帧")
            
            # Reset speech state
            #重置讲话状态，这次话说完了，闭嘴
            is_speaking_was = self.is_speaking
            self.is_speaking = False
            self.silence_started = 0
            
            # Save a copy of speech frames before resetting
            frames_copy = self.speech_frames.copy()
            #先复制录下的所有音频帧到 frames_copy，以免后续被清空。
            self.speech_frames = []
            #清空 self.speech_frames，准备下一次讲话录音。
            
            # Check if we truly had meaningful speech
            if is_speaking_was and speech_duration > 0.5:  # Additional validation
                # Process in a separate thread to not block monitoring
                self.detection_thread = threading.Thread(
                    target=self._save_and_transcribe, 
                    args=(frames_copy,)
                )
                self.detection_thread.daemon = True
                self.detection_thread.start()
                #用 线程 处理保存和转写，这样不会阻塞麦克风监听。
            else:
                print(f"语音太短或者无效: {speech_duration:.2f}秒")
                self.app.update_status("Ready")
        else:
            # Too short, reset without processing
            print(f"语音太短 ({speech_duration:.2f}秒 < {self.min_speech_duration}秒)，忽略")
            self.is_speaking = False
            self.silence_started = 0
            self.speech_frames = []
            self.app.update_status("Ready")




    
    def _save_and_transcribe(self, frames):
        """Save speech frames to file and start transcription"""
        try:
            temp_filename = f"speech_{int(time.time())}.wav"
            #单独的给这个对象命名！生成文件名，拼成 speech_1691847275.wav 这种名字。，独一无二
            print(f"保存语音到 {temp_filename}")
            
            # Ensure the audio object exists检查必要条件
            if not self.audio:
                print("错误: 音频对象不存在，无法保存语音")
                return
            
            # Check if we have frames
            if not frames or len(frames) == 0:
                print("错误: 没有语音帧可以保存")
                return
            
            # Save frames to WAV file 保存音频文件
            wf = wave.open(temp_filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            #writeframes(b''.join(frames)) → 把 frames 列表里的字节拼接成一个完整的音频流，然后一次性写入文件。
            wf.close()
            
            # Verify the file was saved验证文件是否保存成功
            if os.path.exists(temp_filename) and os.path.getsize(temp_filename) > 0:
                print(f"语音文件已保存: {temp_filename}, 大小: {os.path.getsize(temp_filename)} 字节")
            else:
                print(f"保存语音文件失败: {temp_filename}")
                return
            
            # 不再创建占位符，直接发送进行转录
            # 确保UI响应完成后再进入繁重的语音处理
            self.app.after(100, lambda: self._send_for_transcription(temp_filename))
            # 为什么不直接调用？
            # 转录可能是耗时操作（网络请求 / 模型推理），直接调用会卡住 UI。
            # 先让 UI 有时间刷新（比如显示“正在转录”），再开始处理，用户体验更流畅
                        
        except Exception as e:
            error_msg = f"处理语音出错: {e}"
            print(error_msg)
            self.app.update_status(error_msg)





    
    def _send_for_transcription(self, audio_file):
        """Send audio file for transcription after UI is updated"""
        try:
            print(f"发送语音文件进行转写: {audio_file}")
            # Send for transcription - without placeholder ID
            self.app.transcribe_audio(audio_file, priority=True)
            #最为重要的是这个调用主程序的对于转录函数
        except Exception as e:
            error_msg = f"发送转写请求时出错: {e}"
            print(error_msg)
            self.app.update_status(error_msg)





 
class WebcamHandler:
    def __init__(self, app):
        self.app = app
        self.running = False
        self.paused = False  # Flag to indicate if analysis is paused
        self.processing = False  # Flag to indicate if analysis is in progress
        self.cap = None
        self.webcam_thread = None
        self.last_webcam_image = None  # Store the most recent webcam image
        self.debug = True  # Set to True to enable debugging output
        
        # Sequential processing control
        self.analysis_running = False
        
        # Camera window
        self.camera_window = None
    
    def start(self):
        """Start webcam capture process"""
        if not self.running:
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    self.app.update_status("Cannot open webcam")
                    return False
                
                self.running = True
                
                # Create the camera window
                self.create_camera_window()
                
                # Start processing thread
                self.webcam_thread = threading.Thread(target=self._process_webcam)
                self.webcam_thread.daemon = True
                self.webcam_thread.start()
                
                # Start analysis (important - this kicks off the first capture)
                self.analysis_running = True
                
                # Start first analysis after a short delay
                self.app.after(2000, self.trigger_next_capture)
                
                return True
            except Exception as e:
                self.app.update_status(f"Error starting webcam: {e}")
                return False
        return False
    
    def create_camera_window(self):
        """Create a window to display the camera feed"""
        if not self.camera_window or self.camera_window.is_closed:
            self.camera_window = CameraWindow(self.app)
            self.camera_window.title("Camera Feed")
            # Position the window to the right of the main window
            main_x = self.app.winfo_x()
            main_y = self.app.winfo_y()
            self.camera_window.geometry(f"640x480+{main_x + self.app.winfo_width() + 10}+{main_y}")
    
    def stop(self):
        """Stop webcam capture process"""
        self.running = False
        self.analysis_running = False
        if self.cap:
            self.cap.release()
        
        # Close the camera window
        if self.camera_window:
            self.camera_window.destroy()
            self.camera_window = None
    
    def _process_webcam(self):
        """Main webcam processing loop - just keeps the most recent frame"""
        last_ui_update_time = 0
        ui_update_interval = 0.05  # Update UI at 20 fps
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.app.update_status("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # Store the most recent image
                self.last_webcam_image = img
                
                # Update camera window with the current frame
                current_time = time.time()
                if self.camera_window and not self.camera_window.is_closed and current_time - last_ui_update_time >= ui_update_interval:
                    self.camera_window.update_frame(img)
                    last_ui_update_time = current_time
                
                time.sleep(0.03)  # ~30 fps for capture
            except Exception as e:
                error_msg = f"Webcam error: {e}"
                print(error_msg)
                self.app.update_status(error_msg)
                time.sleep(1)  # Pause before retry
    
    def trigger_next_capture(self):
        """Trigger the next capture and analysis cycle"""
        if self.running and self.analysis_running and not self.paused and not self.processing:
            print(f"触发新一轮图像分析 {time.strftime('%H:%M:%S')}")
            self.capture_and_analyze()
    
    def capture_and_analyze(self):
        """Capture screenshots and send for analysis"""
        if self.processing or self.paused:
            return
        
        try:
            self.processing = True
            self.app.update_status("捕捉图像中...")
            
            # Get both analysis screenshots and current display screenshot
            screenshots, current_screenshot = self._capture_screenshots()
            
            # Show immediate feedback with the current screenshot
            if current_screenshot:
                # Generate placeholder ID for tracking
                placeholder_id = f"img_{int(time.time())}"
                
                # Show a placeholder message in the UI while we wait for analysis
                self.app.add_ai_message("正在分析当前画面...", current_screenshot, is_placeholder=True, placeholder_id=placeholder_id)
                
                if self.debug:
                    print(f"已添加图像占位符到UI: {placeholder_id}")
                
                # Process analysis in another thread to keep UI responsive
                analysis_thread = threading.Thread(
                    target=self._analyze_screenshots, 
                    args=(screenshots, current_screenshot, placeholder_id)
                )
                analysis_thread.daemon = True
                analysis_thread.start()
            else:
                print("未能获取有效截图，跳过分析")
                self.processing = False
                # Try again after a short delay
                self.app.after(1000, self.trigger_next_capture)
                
        except Exception as e:
            error_msg = f"捕获/分析出错: {e}"
            print(error_msg)
            self.app.update_status(error_msg)
            self.processing = False
            # Try again after a delay
            self.app.after(2000, self.trigger_next_capture)
    

    def _analyze_screenshots(self, screenshots, current_screenshot, placeholder_id):
        """Analyze screenshots and update UI"""
        try:
            self.app.update_status("正在分析图像...")
            
            # Upload screenshots to OSS
            screenshot_urls = self._upload_screenshots(screenshots)
            
            if screenshot_urls:
                print(f"已上传 {len(screenshot_urls)} 张图片，开始分析")
                
                # Send for analysis and wait for result (blocking)
                analysis_text = self._get_image_analysis(screenshot_urls)
                
                if analysis_text:
                    print(f"分析完成，更新占位符: {placeholder_id}")
                    
                    # Extract behavior type for logging
                    behavior_num, behavior_desc = extract_behavior_type(analysis_text)
                    
                    # Log the behavior
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_message = f"{timestamp}-{behavior_num}-{analysis_text}"
                    logging.info(log_message)
                    print(f"行为记录已保存到日志: {behavior_num}-{behavior_desc}")
                    
                    # *** 修改：在这里直接操作app的observation_history，确保记录被添加 ***
                    current_time = time.time()
                    observation = {
                        "timestamp": current_time,
                        "behavior_num": behavior_num,
                        "behavior_desc": behavior_desc,
                        "analysis": analysis_text
                    }
                    
                    self.app.observation_history.append(observation)
                    print(f"WebcamHandler: 已添加新行为到observation_history: {behavior_num}-{behavior_desc}, 当前长度: {len(self.app.observation_history)}")
                    
                    # Process the image analysis directly 
                    if placeholder_id in self.app.placeholder_map:
                        self.app.update_status("处理分析结果...")
                        self.app.update_placeholder(
                            placeholder_id, 
                            analysis_text, 
                            screenshots=[current_screenshot] if current_screenshot else []
                        )
                    else:
                        print(f"警告: 找不到占位符 {placeholder_id}，无法更新UI")
                else:
                    print("图像分析返回空结果")
            else:
                print("未能上传截图，无法进行分析")
        except Exception as e:
            error_msg = f"分析截图时出错: {e}"
            print(error_msg)
            self.app.update_status(error_msg)
        finally:
            # Important: Mark as not processing and trigger next capture
            self.processing = False
            # Add a slight delay before next capture
            self.app.after(1000, self.trigger_next_capture)

    
    def _get_image_analysis(self, image_urls):
        """Send images to Qwen-VL API and get analysis text"""
        try:
            print("调用Qwen-VL API分析图像...")
            
            messages = [{
                "role": "system",
                "content": [{"type": "text", "text": "详细观察这个人正在做什么。务必判断他属于以下哪种情况：1.认真专注工作, 2.吃东西, 3.用杯子喝水, 4.喝饮料, 5.玩手机, 6.睡觉, 7.其他。分析他的表情、姿势、手部动作和周围环境来作出判断。使用中文回答，并明确指出是哪种情况。"}]
            }]
            
            message_payload = {
                "role": "user",
                "content": [
                    {"type": "video", "video": image_urls},
                    {"type": "text", "text": "这个人正在做什么？请判断他是：1.认真专注工作, 2.吃东西, 3.用杯子喝水, 4.喝饮料, 5.玩手机, 6.睡觉, 7.其他。请详细描述你观察到的内容并明确指出判断结果。"}
                ]
            }
            messages.append(message_payload)
            
            completion = qwen_client.chat.completions.create(
                model="qwen-vl-max",
                messages=messages,
            )
            analysis_text = completion.choices[0].message.content
            print(f"图像分析完成，分析长度: {len(analysis_text)} 字符")
            
            return analysis_text
            
        except Exception as e:
            error_msg = f"Qwen-VL API错误: {e}"
            print(error_msg)
            self.app.update_status(error_msg)
            return None
            
    def toggle_pause(self):
        """Toggle the paused state of the analysis cycle"""
        self.paused = not self.paused
        status = "已暂停分析" if self.paused else "已恢复分析"
        self.app.update_status(status)
        print(status)
        
        # If unpausing, trigger next capture
        if not self.paused and not self.processing:
            self.app.after(500, self.trigger_next_capture)
    
    def get_current_screenshot(self):
        """Get the most recent webcam image"""
        return self.last_webcam_image
    
    def _capture_screenshots(self, num_shots=4, interval=0.1):
        """Capture multiple screenshots from webcam for analysis
           Return both the full set (for analysis) and one current screenshot for display"""
        screenshots = []
        for i in range(num_shots):
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            screenshots.append(img)
            time.sleep(interval)
        
        # Capture one more current frame specifically for display
        ret, current_frame = self.cap.read()
        current_screenshot = None
        if ret:
            current_frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            current_screenshot = Image.fromarray(current_frame_rgb)
        
        if self.debug:
            print(f"已捕获 {len(screenshots)} 张截图用于分析和 1 张当前截图")
            
        return screenshots, current_screenshot
    
    def _upload_screenshots(self, screenshots):
        """Upload screenshots to OSS and return URLs"""
        try:
            auth = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
            bucket = oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET)
            
            if self.debug:
                print(f"正在上传 {len(screenshots)} 张截图到OSS")
                
            oss_urls = []
            for i, img in enumerate(screenshots):
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG')
                buffer.seek(0)
                
                object_key = f"screenshots/{int(time.time())}_{i}.jpg"
                
                result = bucket.put_object(object_key, buffer)
                if result.status == 200:
                    url = f"https://{OSS_BUCKET}.{OSS_ENDPOINT}/{object_key}"
                    oss_urls.append(url)
                    if self.debug:
                        print(f"已上传图片 {i+1}: {url}")
                else:
                    error_msg = f"上传错误，状态码: {result.status}"
                    print(error_msg)
                    self.app.update_status(error_msg)
            
            return oss_urls
        except Exception as e:
            error_msg = f"上传图片时出错: {e}"
            print(error_msg)
            self.app.update_status(error_msg)
            return []



            #上面第一个文件已经有，到时候需要优化

class AudioPlayer:
    #是一个文字转语音 (TTS) 播放器的线程化队列处理系统
    def __init__(self, app):
        self.app = app
        self.current_audio = None
        self.playing = False
        self.play_thread = None
        self.skip_requested = False
        
        # 修改为优先级队列
        self.tts_queue = queue.PriorityQueue()
        #优先级队列：队列中的元素会按照优先级（数字）排序，数字越小优先级越高。
        # 存放格式是 (priority, timestamp, text)
        # priority: 优先级，0 最高，数字越大优先级越低
        # timestamp: 请求进入队列的时间
        # text: 要朗读的文字


        self.tts_thread = None
        self.tts_running = False
        
        # 最大队列长度限制
        self.max_queue_size = 1
    
    def start_tts_thread(self):
        """启动TTS处理线程"""
        if not self.tts_running:
            self.tts_running = True
            self.tts_thread = threading.Thread(target=self._process_tts_queue)
            self.tts_thread.daemon = True
            self.tts_thread.start()
            print("TTS处理线程已启动")
    
    def _process_tts_queue(self):
        #处理队列
        """处理TTS队列中的文本，按优先级播放"""
        while self.tts_running:
            try:
                if not self.tts_queue.empty() and not self.playing:
                    #队列不为空 并且 当前没有正在播放音频时才会取任务
                    # 获取优先级最高的项目 (priority, timestamp, text)
                    priority, timestamp, text = self.tts_queue.get()
                    
                    # 检查是否过期（超过10秒的低优先级消息被视为过期）
                    # 目的是让旧消息不再打断用户当前的操作。
                    current_time = time.time()
                    if priority > 1 and current_time - timestamp > 10:
                        print(f"忽略过期的TTS请求 (已过{current_time - timestamp:.1f}秒): '{text[:30]}...'")
                        self.tts_queue.task_done()
                        #使用queue.task_done()方法通知队列，这样Queue对象就可以知道队列中那一项已经被处理完毕了。
                        continue
                    
                    #表示当前的文本没有过期！可以朗读
                    print(f"从TTS队列获取文本 (优先级: {priority}): '{text[:30]}...'")
                    #播放处理
                    self._synthesize_and_play(text)
                    self.tts_queue.task_done()
                time.sleep(0.1)
            except Exception as e:
                print(f"处理TTS队列时出错: {e}")
                time.sleep(1)
    


    def play_text(self, text, priority=2):
        """将文本添加到TTS队列，支持优先级
           优先级: 1=用户语音回复(最高), 2=图像分析(普通)
        """
        if not text or len(text.strip()) == 0:
            print("警告: 尝试播放空文本，已忽略")
            return
        



        # 清理队列，如果是高优先级请求或队列已满
        if priority == 1 or self.tts_queue.qsize() >= self.max_queue_size:
            #如果是最高优先级（1），清空整个队列（马上播放它）。
            #如果队列满了（max_queue_size 默认 1），丢掉一些旧的任务。
            self._clean_queue(priority)
            #调用下面的清理队列函数



            
        print(f"添加文本到TTS队列 (优先级: {priority}): '{text[:30]}...'")
        
        # 再次确保TTS处理线程已启动
        if not self.tts_running or not self.tts_thread or not self.tts_thread.is_alive():
            self.start_tts_thread()
        
        # 添加到队列（包含优先级和时间戳）
        self.tts_queue.put((priority, time.time(), text))
    
    def _clean_queue(self, new_priority):
        """清理队列，保留更高优先级的项目"""
        if self.tts_queue.empty():
            return
            
        # 如果是最高优先级请求，清空所有正在排队的音频
        if new_priority == 1:
            print("收到高优先级语音请求，清空当前TTS队列")
            #遍历队列，把所有任务取出丢掉
            while not self.tts_queue.empty():
                try:
                    self.tts_queue.get_nowait()
                    #用于从队列中取出一个元素。如果队列为空，它不会阻塞，而是立即抛出一个 queue.Empty 异常。
                    self.tts_queue.task_done()
                except:
                    pass
            return
        
        # 对于普通优先级，仅保持队列在最大长度以下
        while self.tts_queue.qsize() >= self.max_queue_size:
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.task_done()
                print("队列已满，移除最旧的TTS请求")
            except:
                break

    # 之前的方法保持不变...
    def _synthesize_and_play(self, text):
        """合成并播放语音（内部方法，由队列处理器调用）"""
        self.app.update_status("正在合成语音...")
        print(f"TTS合成: '{text}'")
        
        # Set playing status to disable voice detection
        self.app.is_playing_audio = True
        
        try:
            #调用 TTS 引擎
            #是外部的 TTS（Text-to-Speech）合成器 API 封装类。
            synthesizer = SpeechSynthesizer(model=TTS_MODEL, voice=TTS_VOICE)
            audio = synthesizer.call(text)
            
            #空的情况的处理方法
            if audio is None:
                error_msg = "TTS返回空数据，跳过语音播放"
                print(error_msg)
                self.app.update_status(error_msg)
                self.app.is_playing_audio = False
                return
            
            output_file = f'output_{int(time.time())}.mp3'
            with open(output_file, 'wb') as f:
                f.write(audio)
            
            print(f"TTS文件已保存: {output_file}")
            self._play_audio_file_internal(output_file)
            #这个函数就在下面
        except Exception as e:
            error_msg = f"TTS错误: {e}"
            print(error_msg)
            self.app.update_status(error_msg)
            self.app.is_playing_audio = False
    


    def play_audio_file(self, file_path):
        #公共方法，可以播放任意本地音频文件（跳过 TTS 队列）。
        """公共方法用于播放音频文件"""
        print(f"请求播放音频文件: {file_path}")
        
        # 跳过当前播放并等待
        if self.playing:
            self.skip_requested = True
            if self.play_thread and self.play_thread.is_alive():
                print("等待当前播放结束...")
                self.play_thread.join(timeout=2.0)
                
        # 直接播放文件，不通过队列
        self._play_audio_file_internal(file_path)
    



    def _play_audio_file_internal(self, file_path):
        #这是内部方法，不是直接给外部调用的，而是由 TTS 合成完成后调用。
        """内部方法用于实际播放音频文件"""
        print(f"开始播放音频文件: {file_path}")
        
        # 确保之前的播放已停止
        if self.playing:
            self.skip_requested = True
            if self.play_thread and self.play_thread.is_alive():
                self.play_thread.join(timeout=1.0)
        #确保之前的播放已结束
        #如果当前 self.playing 为 True，说明有音频正在播，就设置 self.skip_requested = True，
        # 并等待旧线程结束（join(timeout=1.0)）。
        
        #标记正在播放
        self.skip_requested = False
        self.playing = True
        
        # Mark system as playing audio to disable voice detection
        self.app.is_playing_audio = True
        
        #启动新线程去播放
        self.play_thread = threading.Thread(target=self._play_audio, args=(file_path,))
        self.play_thread.daemon = True
        self.play_thread.start()



    

    def _play_audio(self, file_path):
        #播放线程实际干活的地方。
        """Audio playback worker thread"""
        self.app.update_status("正在播放语音...")
        
        try:
            # Check if file exists
            #检查文件是否存在、大小是否正常
            if not os.path.exists(file_path):
                error_msg = f"音频文件不存在: {file_path}"
                print(error_msg)
                self.app.update_status(error_msg)
                self.playing = False
                self.app.is_playing_audio = False
                return
                
            # Check file size
            file_size = os.path.getsize(file_path)
            print(f"音频文件大小: {file_size} 字节")
            if file_size == 0:
                error_msg = f"音频文件为空: {file_path}"
                print(error_msg)
                self.app.update_status(error_msg)
                self.playing = False
                self.app.is_playing_audio = False
                return
            


            # Load audio file加载 MP3 文件
            try:
                sound = AudioSegment.from_file(file_path, format="mp3")
                #将磁盘上的原始音频文件转换为程序中可操作的、结构化的音频数据对象。
                # pydub 库的 AudioSegment 类，可以加载音频文件。
                #.from_file() 是一个类方法，用于从磁盘上的音频文件（如 MP3, WAV, FLAC 等）创建（加载）一个 AudioSegment 对象。

                print(f"成功加载音频: 长度 {len(sound)/1000:.2f}秒")
            except Exception as e:
                error_msg = f"加载音频失败: {e}"
                print(error_msg)
                self.app.update_status(error_msg)
                self.playing = False
                self.app.is_playing_audio = False
                return
            
            # Play the audio
            try:
                player = play(sound)
                #play() 也是 pydub 提供的播放方法，返回一个线程/进程句柄（player）
                print("音频开始播放")
                

                # Wait until playing is done or skip is requested
                while self.playing and not self.skip_requested:
                    if not player.is_alive():
                        print("音频播放完成")
                        break
                    time.sleep(0.1)
                    
                if self.skip_requested:
                    print("音频播放被跳过")
            except Exception as e:
                error_msg = f"播放时出错: {e}"
                print(error_msg)
                self.app.update_status(error_msg)
                

            # 尝试删除临时文件
            try:
                if os.path.exists(file_path) and file_path.startswith('output_'):
                    os.remove(file_path)
                    #说明合成的 TTS 文件一般是 output_时间戳.mp3，播放完会删除，避免磁盘堆积。
                    print(f"临时文件已删除: {file_path}")
            except Exception as e:
                print(f"删除临时文件出错: {e}")
                
        except Exception as e:
            error_msg = f"音频播放错误: {e}"
            print(error_msg)
            self.app.update_status(error_msg)
        
        self.playing = False
        # Reset playing status to re-enable voice detection
        self.app.is_playing_audio = False
        self.app.update_status("Ready")

    # 命名约定：上述两个最大区别
    # Python 里单下划线 _ 开头的方法是一种“约定”，意思是内部使用，不建议外部直接调用。
    # 没有 _ 的 play_audio_file 是公共接口，给类的外部直接用的。
    # play_audio_file：入口方法，负责一些“外部调用需要的前置处理”。
    # _play_audio_file_internal：核心方法，只管实际播放（启动播放线程），不做额外的外部兼容处理。


    def skip_current(self):
        """Skip the currently playing audio"""
        if self.playing:
            self.skip_requested = True
            self.app.update_status("跳过当前音频...")
            print("已请求跳过当前音频")
            
            # Reset playing status immediately to re-enable voice detection
            self.app.is_playing_audio = False
            #立刻设置 is_playing_audio = False（重新允许语音识别）
            
    def stop(self):
        """停止所有播放和处理"""
        self.skip_current()
        self.tts_running = False
        
        # 清空队列
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
                #立刻取出队列中的一个元素。
                self.tts_queue.task_done()
                #告诉队列**“我刚才取出来的那个任务已经处理完了”**。
            except:
                pass









# ---------------- UI Class ----------------
class MultimediaAssistantApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Set up priority queue for async processing
        self.message_queue = queue.PriorityQueue()
        #优先级消息队列，存放待处理的“任务”（比如语音输入、图像分析结果）。
        self.processing_thread = None
        #后台线程对象处理
        self.processing_running = False
        
        # Message sequence tracking (for updating placeholders)
        self.message_id = 0
        #给消息分配递增的 ID。
        self.placeholder_map = {}  # Maps placeholder IDs to their row indexes
        #记录 UI 中的占位符位置（比如“正在分析当前画面...”所在的行），方便更新。
        
        # 先定义系统消息
        # 为什么可以直接这样写？
        # 因为 chat_context 会在调用 DeepSeek API 时直接传进去：
        #response = deepseek_client.chat.completions.create（...messages=self.chat_context,...）

        self.system_message = {"role": "system", "content": """你是一个监督工作状态的AI助手，负责提高用户的工作效率和健康习惯。

        你需要：
        1. 总是称呼用户为"帆哥！"
        2. 根据观察到的用户行为，分为以下几类并作出相应回应：
        - 如果用户在认真工作：积极鼓励，赞扬他的专注，支持他继续保持
        - 如果用户在喝水：表示赞同，鼓励多喝水保持健康
        - 如果用户在吃东西：严厉批评，提醒工作时间不要吃零食，影响效率和健康
        - 如果用户在喝饮料（非水）：批评他，提醒少喝含糖饮料，建议换成水
        - 如果用户在玩手机：非常严厉地批评，要求立即放下手机回到工作状态
        - 如果用户在打瞌睡/睡觉：大声呵斥，提醒他不要偷懒，建议站起来活动或喝水提神
        - 其他行为：根据是否有利于工作效率来决定态度
        3. 对积极行为（工作、喝水）使用鼓励赞赏的语气
        4. 对消极行为（吃东西、玩手机、喝饮料、睡觉）使用批评或训斥的语气
        5. 每次回应控制在30字以内，简短有力
        6. 语气要根据行为类型明显区分 - 鼓励时温和友好，批评时严厉直接
        7. 非常重要：当用户询问自己的行为时（如"我有没有喝饮料"），你必须查看提供的历史行为记录和统计数据，根据实际历史回答，不要臆测

        记住：你的目标是监督帆哥保持高效工作状态，减少不良习惯！同时准确回答关于他历史行为的问题。
        """}

        #告诉 AI（DeepSeek）它的角色、任务和行为规范，这就是提示词（Prompt）。


        # 然后初始化聊天上下文，使用系统消息
        self.chat_context = [self.system_message]
        #聊天历史上下文，最开始只包含系统消息。
        self.observation_history = []  # 存储历史观察记录，（摄像头分析得出的结果）
        self.behavior_counters = {
            "work": 0,      # 工作计数
            "eating": 0,    # 吃东西计数
            "drinking_water": 0,  # 喝水计数
            "drinking_beverage": 0,  # 喝饮料计数
            "phone": 0,     # 玩手机计数
            "sleeping": 0,  # 睡觉计数
            "other": 0      # 其他行为计数
        }
        #各种行为出现的次数（比如今天喝水了几次、玩手机几次）



        #提醒机制
        self.last_behavior = None  # 上次检测到的行为
        self.continuous_behavior_time = 0  # 持续行为的开始时间

        self.reminder_thresholds = {
            "eating": 2,    # 吃零食提醒阈值
            "drinking_beverage": 2,  # 喝饮料提醒阈值
            "sitting": 30*60,  # 久坐提醒阈值（30分钟）
            "phone": 1,     # 玩手机提醒阈值（次数较低，因为更需要及时制止）
        }
        self.last_reminder_time = {  # 上次提醒时间
            "eating": 0,
            "drinking_beverage": 0,
            "sitting": 0,
            "phone": 0,
            "encouragement": 0  # 鼓励的上次时间
        }
        self.reminder_interval = 10*60  # 两次提醒之间的最小间隔（10分钟）
        self.sitting_start_time = time.time()  # 开始坐下的时间
        

        #其他状态变量
        # Last image analysis for context
        self.last_image_analysis = ""
        
        # Timestamp tracker
        self.last_timestamp = 0
        self.timestamp_interval = 60  # Show timestamp every 60 seconds
        #控制时间戳显示（每隔 60 秒显示一次）。
        
        # Audio playback status to prevent voice detection during playback
        self.is_playing_audio = False
        




        # Setup UI 初始化：定义的函数在下面
        self.setup_ui()
        
        # Initialize system components after UI
        #核心功能组件初始化
        self.audio_recorder = AudioRecorder(self)
        self.webcam_handler = WebcamHandler(self)
        #self.webcam_handler 被初始化为 WebcamHandler 类的实例，
        # 并且传入了当前的 MultimediaAssistantApp 实例（self）。
        # 这意味着 WebcamHandler 类的所有方法都可以通过 self.webcam_handler 访问。
        self.audio_player = AudioPlayer(self)
        self.voice_detector = VoiceActivityDetector(self)
        #为什么“核心功能组件初始化”要在这里才调用？
        #1.统一管理、可维护性好，所有模块都在这里创建，然后可以随时调用
        #2：模块之间需要主应用类的引用（self.app）


        #self.app的作用：
        #1. 访问主应用的属性和方法：
        #  例如，你可以使用 self.app.title 来获取主应用的标题，
        #  或者调用 self.app.update_status("新的状态") 来更新主应用的状态。
        #2. 跨模块通信：
        #  不同的模块（如 UI、音频处理、视频处理等）可以通过 self.app 来进行通信。
        #  例如，当一个模块需要通知主应用某个事件发生时，
        #  可以调用 self.app.some_method() 来触发主应用的相应方法。
        #3. 访问主应用的资源：
        #  例如，你可以使用 self.app.config 来访问主应用的配置信息，


        #理解核心：self.webcam_handler = WebcamHandler(self)为例子：
        # 主程序 → 组件：主程序把自己（self）作为参数传给组件（WebcamHandler），让组件持有一个指向主程序的引用。
        # 组件 → 主程序：组件内部可以通过 self.app 来调用主程序的方法、改主程序的状态、更新 UI。
        # 这样一来就形成了双向沟通：
        # 主程序能控制组件（app.webcam_handler.start()）
        # 组件也能反向通知主程序（self.app.update_status("正在分析图像...")）

        

        #具体的函数都在下面
        # Setup key bindings 绑定键盘快捷键
        self.setup_key_bindings()
        
        # Start background processing启动后台线程
        self.start_processing_thread()
        
        # Start webcam after a short delay延迟启动设备
        self.after(1000, self.start_webcam)
        
        # Start voice monitoring after webcam init
        self.after(2000, self.start_voice_monitoring)
        
        # Start timestamp check
        self.check_timestamp()
        
        # Start audio player TTS thread
        self.after(3000, self.audio_player.start_tts_thread)
        # 1000ms 后：启动摄像头。
        # 2000ms 后：启动语音检测。
        # 立即：启动时间戳检查。
        # 3000ms 后：启动 TTS 线程。
    

    def start_webcam(self):
        """Start webcam capture after UI initialization"""
        if not self.webcam_handler.start():
            self.update_status("Failed to start webcam. Check your camera.")
    
    def start_voice_monitoring(self):
        """Start continuous voice activity detection"""
        self.voice_detector.start_monitoring()
        self.update_status("语音监测已启动")
    
# 为什么可以直接调用：
# 实例化：self.webcam_handler 和 self.voice_detector 在类的构造函数中作为对象被创建并赋值给当前类的实例（self）。
# 访问实例方法：通过 self.webcam_handler 和 self.voice_detector，
# 你可以访问这两个对象的方法（例如 start() 和 start_monitoring()），因为它们已经是当前类实例的一部分。





    def setup_ui(self):
        """Initialize the user interface"""
        self.title("Book思议的结晶")
        #窗口标题随便改：
        self.geometry("1000x800")
        self.default_font_family = "微软雅黑"  
        # 全局默认字体：可以替换为任何你想用的字体，如"Arial", "Times New Roman", "黑体"等
        
        # 定义不同大小的字体
        self.title_font = (self.default_font_family, 16, "bold")
        self.message_font = (self.default_font_family, 12)
        self.name_font = (self.default_font_family, 12, "bold")
        self.status_font = (self.default_font_family, 10)
        self.timestamp_font = (self.default_font_family, 9)
                
        # 配置主窗口的网格布局’
        #让主窗口的第 0 列、第 0 行可以自动拉伸填充整个窗口。
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        #weight=1 表示分配的空间比例，数字越大占得越多。
        #默认情况下，所有列和行的权重都是0，这意味着它们不会根据窗口大小的变化而自动调整大小。


        
        # Create main frame创建主框架（main_frame）
        self.main_frame = ctk.CTkFrame(self)
        #别忘了：CTkFrame 是 CustomTkinter 的容器，相当于一个盒子。
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        #sticky="nsew" 让它在上下左右都对齐。
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=0)
        


        # Create chat display创建聊天显示区
        self.chat_frame = ctk.CTkScrollableFrame(self.main_frame)
        #CTkScrollableFrame：带滚动条的容器，方便显示大量聊天记录。这里放所有对话消息（AI 和用户）。
        self.chat_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.chat_frame.grid_columnconfigure(0, weight=1)
        


        # Create status bar创建状态栏
        self.status_frame = ctk.CTkFrame(self.main_frame)
        self.status_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        #sticky="ew" 表示横向拉伸。
        self.status_frame.grid_columnconfigure(0, weight=1)
        


        # Status label添加状态标签
        self.status_label = ctk.CTkLabel(self.status_frame, text="Ready", anchor="w")
        #显示当前系统状态（默认是“Ready”）。anchor="w"：文字靠左对齐。
        self.status_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        



        # Instruction label添加操作说明标签，CTk自带的功能函数，用以显示
        self.instruction_label = ctk.CTkLabel(
            self.status_frame, 
            text="自动语音检测已启用, 'Space' 跳过语音/暂停分析",
            font=("Arial", 10)
        )
        self.instruction_label.grid(row=0, column=1, padx=10, pady=5, sticky="e")
        #放在状态栏的右边（sticky="e"）。


        # 检查头像图片是否存在
        ai_avatar_path = "ai_avatar.png"  # 在程序目录下放置此图片
        user_avatar_path = "user_avatar.png"  # 在程序目录下放置此图片
        


        # 加载头像（如果本地图片存在则使用本地图片，否则使用生成的圆形）
        self.ai_avatar = self.create_circle_avatar((50, 50), "blue", "DS", image_path=r"E:\沙粒云\自媒体\2025视频制作\20250221deepseekcamera\ds.png")
        self.user_avatar = self.create_circle_avatar((50, 50), "green", "USER", image_path=r"E:\沙粒云\自媒体\2025视频制作\20250221deepseekcamera\user.png")
        #create_circle_avatar() 是自定义方法，内部用 Pillow（PIL）画圆、加文字。
        #优先用本地图片，否则生成一个带文字的圆形头像。



        # Add welcome message
        self.chat_row = 0
        #记录聊天记录的行号，新增消息会按这个计数往下排。
        self.add_ai_message("欢迎使用多模态助手! 我会实时分析摄像头画面并回应。"
                        "系统已启用自动语音检测，直接说话即可。空格键可跳过当前语音播放并暂停/恢复分析。")
        #把欢迎消息显示在聊天框中（左侧，AI 头像）。这个函数后面会有2300多行左右


        # 主窗口
        #  └── main_frame（主框架）
        #      ├── chat_frame（聊天区）
        #      └── status_frame（状态栏）
        #          ├── status_label（状态文字）
        #          └── instruction_label（快捷键说明）

        #辅助理解：grid
        # 主窗口 (grid)
        # ┌─────────────────────────────────────────────┐
        # │ [row=0,col=0] main_frame                    │
        # │   ┌─────────────────────────────────────┐   │
        # │   │ [row=0,col=0] chat_frame                │  ← 聊天内容滚动显示
        # │   ├─────────────────────────────────────┤
        # │   │ [row=1,col=0] status_frame              │  ← 状态栏
        # │   │   ├─ col=0: status_label (左侧状态)  │
        # │   │   └─ col=1: instruction_label (右侧说明)  │
        # │   └─────────────────────────────────────┘
        # └─────────────────────────────────────────────┘











    def create_circle_avatar(self, size, color, text, image_path=None):
        # 功能一句话概括，不太重要可以暂时掠过
        # 生成一个圆形头像，可以：
        # 优先使用本地图片并裁成圆形
        # 如果没图，就画一个彩色圆+文字
        # 最后转成 CustomTkinter 能显示的 CTkImage。

        """创建一个圆形头像，可以使用本地图片或生成带文字的圆形"""
        from PIL import Image, ImageDraw, ImageFont, ImageOps
        
        if image_path and os.path.exists(image_path):
            try:
                # 加载本地图片
                original_img = Image.open(image_path)
                # 调整大小
                original_img = original_img.resize(size, Image.LANCZOS)
                
                # 创建一个透明的圆形遮罩
                mask = Image.new('L', size, 0)
                draw = ImageDraw.Draw(mask)
                draw.ellipse((0, 0, size[0], size[1]), fill=255)
                
                # 将图片裁剪成圆形
                img = Image.new('RGBA', size, (0, 0, 0, 0))
                img.paste(original_img, (0, 0), mask)
                
                # 转换为CTkImage
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=size)
                return ctk_img
                
            except Exception as e:
                print(f"加载头像图片出错: {e}, 使用默认头像")
                # 如果图片加载失败，回退到默认头像
                pass
        
        # 如果没有提供图片路径或加载失败，生成默认头像
        img = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # 绘制圆形
        cx, cy = size[0] // 2, size[1] // 2
        radius = min(cx, cy) - 2
        
        if color == "blue":
            fill_color = (0, 100, 200, 255)
        elif color == "green":
            fill_color = (0, 150, 100, 255)
        else:
            fill_color = (100, 100, 100, 255)
        
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill_color)
        
        # 添加文字
        try:
            font = ImageFont.truetype("arial.ttf", size=radius // 2)
        except IOError:
            font = ImageFont.load_default()
        
        text_width, text_height = draw.textsize(text, font=font) if hasattr(draw, 'textsize') else (radius, radius//2)
        draw.text((cx - text_width // 2, cy - text_height // 2), text, fill="white", font=font)
        
        # 转换为CTkImage
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=size)
        return ctk_img





    
    def setup_key_bindings(self):
        """Set up keyboard shortcuts"""
        #功能： 设置键盘快捷键，让用户能快速触发录音、停止录音、跳过音频等功能。：主动干预
        self.bind("<r>", lambda e: self.start_voice_recording())
        self.bind("<s>", lambda e: self.stop_voice_recording())
        self.bind("<space>", lambda e: self.skip_audio())
        #self.bind("<键>", 函数)：绑定窗口内部快捷键。
        # 这里用 lambda e: ... 是因为 Tkinter 绑定的回调默认会接收一个事件参数 e。
        # 举例：
        # 按 r → 调用 start_voice_recording()
        # 按 s → 调用 stop_voice_recording()
        # 按 空格 → 调用 skip_audio()

        #全局快捷键（窗口不激活也能用）
        # Also add keyboard module hotkeys for global control
        keyboard.add_hotkey('r', self.start_voice_recording)
        keyboard.add_hotkey('s', self.stop_voice_recording)
        keyboard.add_hotkey('space', self.skip_audio)
        #keyboard 模块可以监听系统全局按键，即使你点到其他程序也能触发。


        # self.bind() → 窗口级别，只有程序窗口在前台时有效。
        # keyboard.add_hotkey() → 全局级别，无论窗口是否在前台都有效。

    

    def start_processing_thread(self):
        #功能： 启动一个后台线程，专门处理消息队列。
        """Start the background message processing thread"""
        #必须记住四板斧：
        #设置状态->创建一个线程指定后台函数->建立守护线程（主程序退出自动关）->启动线程
        self.processing_running = True
        self.processing_thread = threading.Thread(target=self.process_message_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    


    def process_message_queue(self):
        #功能： 循环从优先级队列取出消息并处理。
        """Process messages from the priority queue"""
        #存储内容是 (priority, msg_id, message)。
        while self.processing_running:
            try:
                if not self.message_queue.empty():
                    # Get message with priority (lower number = higher priority)
                    priority, msg_id, message = self.message_queue.get()
                    #→ 取出一条消息（阻塞式或非阻塞式）
                    print(f"处理消息: 类型={message['type']}, 优先级={priority}, ID={msg_id}")
                    self.handle_message(message, msg_id)
                    #→ 根据消息类型做不同的事（比如显示文本、播放音频等）。函数就在下面
                    self.message_queue.task_done()#告诉队列这条任务已完成。
                time.sleep(0.05)  # Reduced sleep for more responsiveness，减少CPU占用

            #异常处理
            except Exception as e:
                error_msg = f"Error processing message: {e}"
                print(error_msg)
                self.update_status(error_msg)



    
    def handle_message(self, message, msg_id=None):
        #从队列里取到一条消息后，根据它的 type（类型）判断是图像分析结果还是语音输入，再分别调用对应的处理方法。
        """Handle different message types from the queue"""
        try:

            ##图像分析类消息（type == "image_analysis"）
            if message["type"] == "image_analysis":
            # message 是一个字典，里面有至少两个关键信息：
            # "type" → 消息类别（决定走哪个分支）。
            # "content" → 具体内容（分析结果、语音文本等）。

                # Check if this is a placeholder update
                if "placeholder_id" in message and message["placeholder_id"]:
                    placeholder_id = message["placeholder_id"]
                    print(f"更新图像分析占位符: {placeholder_id}")
                    self.update_placeholder(
                        placeholder_id, 
                        message["content"], 
                        screenshots=message.get("screenshots", [])
                    )
                    #这个函数就在下面：
                else:
                    print(f"处理新图像分析")
                    self.process_image_analysis(
                        message["content"], 
                        message.get("urls", []), 
                        message.get("screenshots", [])
                    )
                    #这个函数也在下面：

            elif message["type"] == "voice_input":
                print(f"处理语音输入: {message['content']}")
                self.process_voice_input(
                    message["content"],
                    placeholder_id=message.get("placeholder_id")
                )
        except Exception as e:
            error_msg = f"处理消息时出错: {e}"
            print(error_msg)
            self.update_status(error_msg)
     
            #关于上述的"placeholder_id"问题：视觉化示意：ID 就是 placeholder_id，用来匹配和替换。
            # (1) 收到占位消息
            # UI:
            #   [ID=42] AI: 正在分析图片...

            # (2) 收到结果消息（带 placeholder_id=42）
            # UI:
            #   [ID=42] AI: 画面中有两只猫在玩耍

            # ┌────────────────────────────┐
            # │ process_message_queue()    │ ← 不断取出消息
            # └─────────────┬──────────────┘
            #               │
            #               ▼
            #      ┌──────────────────────┐
            #      │ handle_message()     │
            #      └─────────┬────────────┘
            #                │
            #       判断 message["type"]
            #      ┌─────────┴─────────┐
            #      │                   │
            # image_analysis       voice_input
            #      │                   │
            #   ┌──┴───┐          ┌────┴────┐
            #   │有占位 │          │ 调用     │
            #   │符 ID  │         │ process  │
            #   │      │          │ _voice_  │
            #   └──┬───┘          │ input()  │
            #      │              └───────——─┘
            # update_placeholder()
            #      │
            #      ▼
            # process_image_analysis()没有占位ID









    def update_placeholder(self, placeholder_id, new_content, screenshots=None):
        """Update a placeholder message with actual content"""
        #用来把 UI 里“占位的临时文字”（例如“正在分析当前画面...”）替换成真正的分析结果，并根据结果让 AI 回复，还可以播语音。
        print(f"更新占位符: {placeholder_id} 内容长度: {len(new_content)}")
        if placeholder_id in self.placeholder_map:
            print(f"找到占位符在位置: {self.placeholder_map[placeholder_id]}")
            

            # Actually replace the placeholder with real content
            if placeholder_id.startswith("img_"):
                #用来区分这是图像分析的结果，而不是语音的
                # This is an image analysis placeholder - add the real analysis
                print(f"添加图像分析结果到UI: {new_content[:50]}...")
                
                # Store the analysis for context
                self.last_image_analysis = new_content
                
                # Find the old row number
                row_num = self.placeholder_map[placeholder_id]
                #💡 先拿到占位符所在行号。
                #self.placeholder_map 是个字典：
                # {
                #     "img_123": 5,  # 第5行是图像分析占位
                #     "voice_456": 8 # 第8行是语音输入占位
                # }



                # Get the frame within the chat_frame at that row
                #很抽象这里：
                #注意：self.chat_frame = ctk.CTkScrollableFrame(self.main_frame)

                for widget in self.chat_frame.winfo_children():
                #winfo_children() 会返回聊天框里的所有“行”对应的控件（其实是一个个 frame，每一行就是一个小框）。
                #grid_info()['row']就是用来查控件当前在第几行的。
                    if int(widget.grid_info()['row']) == row_num:
                        frame = widget
                        #如果行号等于 row_num（比如 2），说明找到了那一行。
                        # Find the text label within the frame
                        for child in frame.winfo_children():
                            # 每一行 frame 里可能有很多子控件（头像、名字、文字等），这里要找到：
                            # 类型是 CTkLabel（文本控件）
                            # 文字内容等于 "正在分析当前画面..."
                            #描述 isinstance () 函数来判断一个对象是否是一个已知的类型
                            #Tkinter中如何获取控件属性的三种方法：使用.cget() 方法、使用.config() 方法和直接访问属性
                            if isinstance(child, ctk.CTkLabel) and child.cget("text") == "正在分析当前画面...":
                                # Update the label text
                                # 把文字更新成分析结果
                                #new_content 就是摄像头分析的结果文字
                                child.configure(text=f"📷 {new_content}")
                                # Change the appearance from placeholder to normal
                                frame.configure(fg_color=("#EAEAEA", "#2B2B2B"))
                                child.configure(text_color=("black", "white"))
                                print(f"成功更新占位符内容")
                                #让它从“占位灰色”变成“正式内容颜色”。
                                break

                            #new_content从何而来？到底是什么？
                            # 来自 handle_message() → 当消息类型是 "image_analysis" 且带 placeholder_id 时：
                                # self.update_placeholder(
                                #     placeholder_id, 
                                #     message["content"],  # 这里就是 new_content
                                #     screenshots=message.get("screenshots", [])
                                # )
                            # 所以 new_content = message["content"]，而 message["content"] 是消息队列里推送的分析结果。
                            #好，那么handle_message(self, message, msg_id=None)里面的message又是从哪里来的？
                            #在 process_message_queue() 里有：
                                # priority, msg_id, message = self.message_queue.get()
                                # print(f"处理消息: 类型={message['type']}, 优先级={priority}, ID={msg_id}")
                                # self.handle_message(message, msg_id)，所以收到的是千问的回复！



                # Extract behavior type for logging记录和行为提取
                behavior_num, behavior_desc = extract_behavior_type(new_content)
                
                # Now generate an AI response based on the analysis
                try:
                    # 调用 AI 生成回复
                    print("调用DeepSeek生成回应...")
                    messages = [
                        self.system_message,
                        {"role": "user", "content": f"基于这个观察: {new_content}, 根据检测到的行为类型给出相应回应。如果是工作或喝水，给予鼓励；如果是吃东西、玩手机、喝饮料或睡觉，给予批评和提醒."}
                    ]
                    
                    response = deepseek_client.chat.completions.create(
                        model="deepseek-chat",
                        messages=messages,
                        stream=False
                    )
                    assistant_reply = response.choices[0].message.content
                    #为什么这样回复：？
                    # DeepSeek 的返回结构是个 JSON，choices 是一个列表（可能有多条回复）。
                    # choices[0] → 取第一条回复（一般只返回一条）。
                    # .message.content → 取这条回复的文本内容。
                    # 这样才能拿到干净的字符串放到 UI 上。

                    print(f"DeepSeek回应: {assistant_reply}")
                    
                    #在聊天框显示 AI 回复。具体函数在比较下面
                    self.add_ai_message(assistant_reply)
                    
                    #用 audio_player 把文字转成语音播放。
                    self.audio_player.play_text(assistant_reply)
                except Exception as e:
                    error_msg = f"DeepSeek API错误: {e}"
                    print(error_msg)
                    self.update_status(error_msg)
                
                # Remove the placeholder from our tracking map
                del self.placeholder_map[placeholder_id]
            
            elif placeholder_id.startswith("voice_"):
                # This is a voice input placeholder - we'll handle in process_voice_input
                pass
    

                # update_placeholder(占位符ID, 新内容)
                # │
                # ├─ 检查 ID 是否在占位符字典 self.placeholder_map
                # │
                # ├─ 如果是图像分析占位符 (img_ 开头)
                # │   ├─ 找到 UI 上对应的控件（label）
                # │   ├─ 替换文字（“正在分析...” → “📷 结果文字”）
                # │   ├─ 修改颜色（占位灰色 → 正常颜色）
                # │   ├─ 保存最新图像分析到 self.last_image_analysis
                # │   ├─ 从结果提取行为类型（extract_behavior_type）
                # │   ├─ 调用 DeepSeek API 生成 AI 回复
                # │   ├─ 把 AI 回复显示到聊天框
                # │   ├─ 播放 AI 回复的语音
                # │   └─ 从占位符字典里删除这个 ID
                # │
                # └─ 如果是语音输入占位符 (voice_ 开头)
                #     └─ 暂时不处理（process_voice_input 会处理）











    def process_voice_input(self, text, placeholder_id=None):
        # 作用概览
        # process_voice_input(self, text, placeholder_id=None)
        # 作用：处理一次用户的语音输入，把它当作对话内容交给 AI（DeepSeek）生成回应，同时结合历史行为记录，让回答更有上下文。
        # 你可以理解为：
        # 用户说了一句话（通过语音识别转成 text）
        # 系统会：
        # 暂停当前语音播放
        # 把用户说的话显示到 UI
        # 分析用户是不是在问某个特定行为（喝饮料、吃东西、玩手机等）
        # 从历史记录里找相关的行为数据
        # 拼成一个带上下文的总结 context_summary
        # 调用 DeepSeek API，得到回复 assistant_reply
        # 把回复显示到 UI 并播放出来

        #逻辑比较长，好长啊/。。。。
        """Process voice input and generate AI response with historical context"""
        print(f"处理语音输入: '{text}'")
        

        # 调试信息：检查observation_history的内容
        print(f"当前observation_history长度: {len(self.observation_history)}")
        for i, obs in enumerate(self.observation_history):
            print(f"记录[{i}]: 行为={obs['behavior_num']}-{obs['behavior_desc']}, 时间={datetime.fromtimestamp(obs['timestamp']).strftime('%H:%M:%S')}")
        

        # 跳过当前音频播放
        print("打断当前语音播放")
        self.audio_player.skip_current()
        


        # 临时禁用语音检测 打断当前语音播放
        was_playing_audio = self.is_playing_audio
        #was_playing_audio 记录之前的状态；过去时嘛！
        self.is_playing_audio = True
        #一般现在时
        
        # 记录语音处理开始时间
        voice_start_time = time.time()
        
        # 添加用户消息到UI， UI 显示用户的这句话
        self.add_user_message(text)
        

        # 定义行为映射表
        behavior_map = {
            "1": "认真专注工作",
            "2": "吃东西",
            "3": "用杯子喝水",
            "4": "喝饮料",
            "5": "玩手机",
            "6": "睡觉",
            "7": "其他"
        }
        
        # 创建行为统计摘要，久坐时间估算
        sitting_duration = time.time() - self.sitting_start_time if self.sitting_start_time > 0 else 0
        
        # 检查用户是不是在问特定行为（这他妈是啥玩意儿？）别忘记了
        #any() 是 Python 内置函数，作用是：判断一个可迭代对象（比如列表、元组、生成器等）中是否有至少一个元素为 True。
        #如果有一个或多个元素为 True，返回 True； 所有元素都为 False，返回 False。
        #(keyword in text for keyword in ["..."])：
        # 生成器表达式（一种简化的循环写法），等价于一个 “临时的循环判断”：
        # 遍历列表 ["有没有喝饮料", "喝饮料了吗", ...] 中的每个元素（给它起个临时名字 keyword），
        # 逐个判断 keyword 是否在 text 里（即 keyword in text）。
        #综上： 意思是：“只要列表中任何一个关键词出现在 text 里，就返回 True”。
        is_asking_about_beverage = any(keyword in text for keyword in ["有没有喝饮料", "喝饮料了吗", "喝了什么", "喝过饮料"])
        is_asking_about_eating = any(keyword in text for keyword in ["有没有吃东西", "吃东西了吗", "吃了什么", "吃过东西"])
        is_asking_about_phone = any(keyword in text for keyword in ["有没有玩手机", "玩手机了吗", "用过手机"])
        is_asking_about_behavior = is_asking_about_beverage or is_asking_about_eating or is_asking_about_phone or "我做了什么" in text
        
        # 建议：太人机了，改！
        # 中文口语表达很多，建议把 text 做简易规整（去空格/同义词表/小写化对英文），或用更鲁棒的意图识别。
        # 关键词可以集中到一个配置里，避免散落在代码中。



        # 创建相关行为的详细记录
        relevant_history = []
        behavior_filter = None
        #behavior_filter 是你要找的行为编号（用于筛选历史）。
        
        if is_asking_about_beverage:
            behavior_filter = "4"  # 喝饮料的行为编号
            print("检测到用户询问饮料相关行为")
        elif is_asking_about_eating:
            behavior_filter = "2"  # 吃东西的行为编号
            print("检测到用户询问进食相关行为")
        elif is_asking_about_phone:
            behavior_filter = "5"  # 玩手机的行为编号
            print("检测到用户询问手机相关行为")
        
        # *** 修改：如果behavior_counters显示有相关行为，但observation_history为空，则从日志文件恢复 ***
        # 历史缺失时，用计数器补一条恢复记录
        #如果用户明确问某类行为，但 observation_history 完全为空，而计数器又显示今天其实发生过这种行为，
        # 就**造一条“恢复记录”**补进去（时间戳设为当前时间-5分钟，方便 AI 有材料可说）。
        if behavior_filter and len(self.observation_history) == 0:
            #当用户明确要查某个行为（behavior_filter 存在），但历史记录列表是空的，就执行下面的逻辑。
            # 先检查行为计数器
            behavior_key = {
                "2": "eating",
                "3": "drinking_water", 
                "4": "drinking_beverage",
                "5": "phone",
                "6": "sleeping"
            }.get(behavior_filter, "other")
            #当用户明确要查某个行为（behavior_filter 存在），但历史记录列表是空的，就执行下面的逻辑。
            #.get(...)的作用：如果behavior_filter不在字典的键里（比如是 "99"），就默认返回 "other"（其他行为）。

            # 如果行为计数器显示有这个行为，但observation_history为空，则添加一个恢复记录
            if self.behavior_counters.get(behavior_key, 0) > 0:
                #self.behavior_counters：是一个计数器字典，
                # 记录每种行为发生过多少次（比如{"eating": 3, "phone": 2}表示吃饭 3 次、用手机 2 次）。
                #这句话的意思：检查用户要查询的这种行为（比如吃饭），计数器里是否有记录（次数 > 0）。
                # 如果有，就说明 "虽然历史记录丢了，但系统确实检测到过这种行为"
                print(f"检测到计数器显示存在{behavior_key}行为，但observation_history为空，添加恢复记录")
                behavior_desc = behavior_map.get(behavior_filter, "未知行为")
                
                # 创建恢复记录
                recovery_observation = {
                    "timestamp": time.time() - 300,  # 假设发生在5分钟前
                    "behavior_num": behavior_filter,
                    "behavior_desc": behavior_desc,
                    "analysis": f"系统检测到用户在从事{behavior_desc}活动（从行为计数器恢复的记录）"
                }
                self.observation_history.append(recovery_observation)
                print(f"已从行为计数器恢复记录：{behavior_filter}-{behavior_desc}")
        



        # 如果用户问了特定行为：在历史中倒序查找匹配
        if behavior_filter:
            print(f"搜索历史记录中的行为编号: {behavior_filter}")
            for obs in reversed(self.observation_history):
                print(f"比较: {obs['behavior_num']} vs {behavior_filter}, 类型: {type(obs['behavior_num'])} vs {type(behavior_filter)}")
                if str(obs['behavior_num']) == str(behavior_filter):  # 确保类型一致
                    #做了 str(...)==str(...) 是为了避免 behavior_num 有时是数字有时是字符串导致不相等。
                    #建议：源头统一 behavior_num 类型（比如始终字符串），可以省去每次比较都转型。
                    obs_time = datetime.fromtimestamp(obs["timestamp"]).strftime("%H:%M:%S")
                    relevant_history.append(f"- {obs_time}: {obs['behavior_desc']} - {obs['analysis'][:150]}...")
                    print(f"找到匹配记录: {obs_time}")
            
            #没找到怎么办？看计数器兜底：
            if not relevant_history:
                # *** 修改：检查行为计数器 ***
                behavior_key = {
                    "2": "eating",
                    "3": "drinking_water", 
                    "4": "drinking_beverage",
                    "5": "phone",
                    "6": "sleeping"
                }.get(behavior_filter, "other")
                
                if self.behavior_counters.get(behavior_key, 0) > 0:
                    # 如果计数器显示有这个行为，但没有找到记录，添加基于计数器的回复
                    relevant_history.append(f"根据系统记录，用户今天有过{behavior_map.get(behavior_filter, '未知')}行为（从行为计数器推断）")
                    print(f"未找到行为编号为{behavior_filter}的历史记录，但计数器显示有这个行为")
                else:
                    relevant_history.append(f"未在历史记录中找到相关的'{behavior_map.get(behavior_filter, '未知')}'行为")
                    print(f"未找到行为编号为{behavior_filter}的历史记录")
        


        # 创建最近行为的详细记录-摘要（最多5条）
        recent_observations = []
        for obs in reversed(self.observation_history[-5:]):
            obs_time = datetime.fromtimestamp(obs["timestamp"]).strftime("%H:%M:%S")
            behavior_desc = obs["behavior_desc"]
            analysis_brief = obs["analysis"][:100] + ("..." if len(obs["analysis"]) > 100 else "")
            recent_observations.append(f"- {obs_time}: {behavior_desc} - {analysis_brief}")
        
        recent_observations_text = "\n".join(recent_observations)
        if not recent_observations:
            recent_observations_text = "没有最近的行为记录"
        

        # 添加最后一次观察的完整内容
        last_observation = ""
        if self.observation_history:
            last_obs = self.observation_history[-1]
            last_time = datetime.fromtimestamp(last_obs["timestamp"]).strftime("%H:%M:%S")
            last_observation = f"最后一次观察 ({last_time}):\n{last_obs['analysis']}"
        else:
            last_observation = "没有观察记录"
        

        # 构建上下文信息，包括特定行为查询结果
        #组装上下文摘要（给 AI 的“前情提要”）
        context_summary = f"""
    用户当前行为统计：
    - 工作: {self.behavior_counters['work']}次
    - 喝水: {self.behavior_counters['drinking_water']}次
    - 吃东西: {self.behavior_counters['eating']}次
    - 喝饮料: {self.behavior_counters['drinking_beverage']}次 {'(检测到用户询问此行为)' if is_asking_about_beverage else ''}
    - 玩手机: {self.behavior_counters['phone']}次 {'(检测到用户询问此行为)' if is_asking_about_phone else ''}
    - 久坐时间: {int(sitting_duration/60)}分钟
    """
        #把统计、特定行为历史、最近记录、最后一次观察拼到一起。这是 prompt 的关键部分，让模型“知道你最近都干了啥”。
        # 如果询问特定行为，添加相关历史记录
        if is_asking_about_behavior and relevant_history:
            context_summary += f"""
    相关行为历史记录:
    {chr(10).join(relevant_history)}

    """
        
        # 添加最近观察记录
        context_summary += f"""
    最近的行为记录:
    {recent_observations_text}

    {last_observation}
    """
        
        # 将用户问题添加到聊天上下文
        user_message = {"role": "user", "content": f"{context_summary}\n\n用户说: {text}"}
        self.chat_context.append(user_message)
        
        # 限制上下文长度
        if len(self.chat_context) > 20:
            self.chat_context = [self.chat_context[0]] + self.chat_context[-19:]
        
        try:
            print(f"调用DeepSeek生成回应，消息历史长度: {len(self.chat_context)}")
            



            # 使用完整的对话历史发送请求，调用 DeepSeek 生成回复 & 统计耗时
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=self.chat_context,
                stream=False
            )
            assistant_reply = response.choices[0].message.content
            print(f"DeepSeek回应: {assistant_reply}")
            
            # 记录语音处理结束时间
            voice_end_time = time.time()
            print(f"语音处理总耗时: {voice_end_time - voice_start_time:.2f}秒")
            
            # 将AI回应添加到对话历史
            assistant_message = {"role": "assistant", "content": assistant_reply}
            self.chat_context.append(assistant_message)
            
            # 添加AI回应到聊天记录
            self.add_ai_message(assistant_reply)
            
            # 使用高优先级播放回复
            self.audio_player.play_text(assistant_reply, priority=1)
        except Exception as e:
            error_msg = f"DeepSeek API error: {e}"
            print(error_msg)
            self.update_status(error_msg)
            # 恢复原来的语音检测状态
            self.is_playing_audio = was_playing_audio

        # 小例子：两条典型输入会发生什么
        # A. 用户问：“我今天喝饮料了吗？”
        # 匹配到 is_asking_about_beverage=True → behavior_filter="4"
        # 去 observation_history 找 behavior_num=="4" 的记录；没找到则：
        # 如果 behavior_counters['drinking_beverage']>0 → 造一条恢复记录 + 在 relevant_history 写 “根据系统记录…（计数器推断）”
        # 否则写 “未在历史记录中找到…”
        # 组装 context_summary（统计、相关历史、最近 5 条、最后一次观察）
        # 发送给 DeepSeek，得到“有/没有 + 细节建议”等回复
        # UI 显示 + TTS 播放
        # B. 用户随便聊天：“今天有点困…”
        # 不触发特定行为查询（behavior_filter=None）
        # 只拼统计 + 最近记录 + 最后一次观察
        # DeepSeek 基于上下文给出建议（比如提醒休息/喝水等）
        # UI 显示 + TTS 播放

        #     常见坑与改进建议
        # 键名一致性
        # behavior_counters 的键必须和映射里一致（work/drinking_water/eating/drinking_beverage/phone/sleeping）。
        # 类型一致
        # behavior_num 建议统一为字符串，避免每次比较还要 str(...)。
        # 状态恢复
        # self.is_playing_audio 成功路径不恢复应由 TTS 结束回调统一处理；否则可能一直处于“播放中”状态，影响 VAD/ASR。
        # 关键词识别
        # 以表驱动/正则/小模型意图分类替代硬编码 in；加入更多近义词。
        # 上下文长度
        # 20 条是经验值；更稳妥是按 token 估算，或把“最近行为摘要”作为独立字段传给系统消息，减少重复。




    def process_image_analysis(self, analysis_text, image_urls, screenshots, placeholder_id=None):
        #，核心功能是处理图像分析结果，并根据分析到的用户行为（比如工作、吃饭、玩手机等）进行跟踪、记录，
        # 最终生成 AI 回应（甚至可能通过语音播放）。可以理解为一个 “行为监测与智能反馈系统” 的核心处理逻辑。
        #接收了五个参数：
        # self：类的实例本身（访问类的变量和方法）；
        # analysis_text：图像分析后的文本结果（比如 “用户正在玩手机”）；
        # image_urls：分析的图片 URL（可能没用上）；
        # screenshots：截图数据（用于在 UI 显示）；
        # placeholder_id：UI 中临时占位符的 ID（用于更新显示）。


        """Process image analysis results, track behavior patterns, and generate context-aware AI response"""
        print(f"处理图像分析: 分析长度 {len(analysis_text)} 字符, 占位符ID: {placeholder_id}")
        
        # 提取行为类型
        behavior_num, behavior_desc = extract_behavior_type(analysis_text)
        #调用 extract_behavior_type 函数（未展示），从分析文本中提取两个关键信息：以恶搞是行为编号，一个是行为描述
        
        # 记录到日志
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"{timestamp}-{behavior_num}-{analysis_text}"
        logging.info(log_message)
        print(f"行为记录已保存到日志: {behavior_num}-{behavior_desc}")
        
        # 存储观察记录
        current_time = time.time()
        observation = {
            "timestamp": current_time,
            "behavior_num": behavior_num,  # 确保这里存储的是字符串类型
            "behavior_desc": behavior_desc,
            "analysis": analysis_text
        }
        
        # 将观察添加到历史记录，保留最近20条
        self.observation_history.append(observation)
        #为什么这么做？：方便后续查询 “用户最近做了什么”，比如用户问 “我刚才在干嘛”，就可以从这个列表里找答案。
        
        # 调试信息：确认添加成功
        print(f"已添加新行为到observation_history: {behavior_num}-{behavior_desc}, 当前长度: {len(self.observation_history)}")
        
        if len(self.observation_history) > 20:
            self.observation_history.pop(0)  # 保留最近20条
            

        # 更新行为计数器
        behavior_map = {
            "1": "work",
            "2": "eating",
            "3": "drinking_water",
            "4": "drinking_beverage",
            "5": "phone",
            "6": "sleeping",
            "7": "other"
        }
        current_behavior = behavior_map.get(behavior_num, "other")
        self.behavior_counters[current_behavior] += 1
        #self.behavior_counters 是一个字典（比如 {"eating":3, "phone":5}）
        print(f"行为计数更新: {current_behavior} = {self.behavior_counters[current_behavior]}")
        



        #  跟踪持续行为（计算行为持续时间）
        if self.last_behavior == current_behavior:# 如果当前行为和上一次相同
            behavior_duration = current_time - self.continuous_behavior_time # 计算持续时间（秒）
        else:# 如果是新行为
            self.continuous_behavior_time = current_time# 重置持续时间计时
            behavior_duration = 0
            #作用：记录用户同一行为持续了多久（比如 “持续工作了 30 分钟”“持续玩手机 15 分钟”）。
        
        self.last_behavior = current_behavior
        



        #跟踪坐姿时间（健康管理）
        # 如果是新的坐姿行为（不是站起来活动），更新坐姿开始时间
        if current_behavior not in ["other"]:  # 假设"other"可能包括站起来活动
            # 如果之前没有记录坐姿开始时间，记录当前时间
            if self.sitting_start_time == 0:
                self.sitting_start_time = current_time
        else:
            # 重置坐姿计时器
            self.sitting_start_time = 0
        #逻辑：如果当前行为不是 “other”（假设 “other” 是站立活动），就认为用户坐着，开始计时；如果是 “other”，说明用户站起来了，重置计时。
        



        # 判断是否需要提醒（健康/效率管理）
        ## 计算当前坐姿持续时间
        sitting_duration = current_time - self.sitting_start_time if self.sitting_start_time > 0 else 0
        should_remind = False# 是否需要提醒（默认不需要）
        reminder_type = None# 提醒类型（比如"eating"、"phone"）
        

        # 判断是否需要提醒：所有的逻辑判断类似：非常重要
        # 核心条件（以吃零食为例）：
        # 当前行为是 “eating”；
        # 吃的次数超过了阈值（reminder_thresholds["eating"]，比如 3 次）；
        # 距离上次提醒的时间超过了间隔（reminder_interval，比如 10 分钟）—— 避免频繁提醒。
        if current_behavior == "eating" and self.behavior_counters["eating"] >= self.reminder_thresholds["eating"] and \
        current_time - self.last_reminder_time["eating"] > self.reminder_interval:
            should_remind = True
            reminder_type = "eating"
            self.last_reminder_time["eating"] = current_time
        
        elif current_behavior == "drinking_beverage" and self.behavior_counters["drinking_beverage"] >= self.reminder_thresholds["drinking_beverage"] and \
            current_time - self.last_reminder_time["drinking_beverage"] > self.reminder_interval:
            should_remind = True
            reminder_type = "drinking_beverage"
            self.last_reminder_time["drinking_beverage"] = current_time
        
        elif current_behavior == "phone" and self.behavior_counters["phone"] >= self.reminder_thresholds["phone"] and \
            current_time - self.last_reminder_time["phone"] > self.reminder_interval:
            should_remind = True
            reminder_type = "phone"
            self.last_reminder_time["phone"] = current_time
        
        elif sitting_duration > self.reminder_thresholds["sitting"] and \
            current_time - self.last_reminder_time["sitting"] > self.reminder_interval:
            should_remind = True
            reminder_type = "sitting"
            self.last_reminder_time["sitting"] = current_time
        


        #  判断是否需要鼓励（正向激励）
        #作用：对 “好行为” 进行鼓励（比如 “工作认真，继续加油”“多喝水对健康好”）
        # 触发条件：
        # 持续工作超过 10 分钟（10*60秒）；
        # 正在喝水；
        # 距离上次鼓励超过间隔（避免频繁鼓励）。
        should_encourage = False
        if (current_behavior == "work" and behavior_duration > 10*60) or \
        (current_behavior == "drinking_water") and \
        (current_time - self.last_reminder_time["encouragement"] > self.reminder_interval):
            should_encourage = True
            self.last_reminder_time["encouragement"] = current_time
        
        # 存储最新的图像分析作为上下文
        self.last_image_analysis = analysis_text
        
        # 添加图像分析到聊天记录
        #调用 add_ai_message 方法，把分析结果显示到 UI 上（如果有截图，就一起显示）。
        if not placeholder_id or placeholder_id not in self.placeholder_map:
            if screenshots and len(screenshots) > 0:
                print(f"添加新的图像分析到UI，带截图")
                self.add_ai_message(f"📷 {analysis_text}", screenshots[0], placeholder_id=placeholder_id)
            else:
                print(f"添加新的图像分析到UI，无截图")
                self.add_ai_message(f"📷 {analysis_text}", placeholder_id=placeholder_id)
        
        # 根据分析结果构建提示
        #构建 AI 回应的提示指令
        #给 AI 模型（DeepSeek）一个 “指令”，告诉它应该怎么回应用户（比如 “要批评” 还是 “要鼓励”）。
        prompt_instruction = ""
        if should_remind:
            if reminder_type == "eating":
                prompt_instruction = "用户持续吃零食，请严厉批评并提醒他工作时间不要吃零食，会影响效率和健康。"
            elif reminder_type == "drinking_beverage":
                prompt_instruction = "用户经常喝饮料（非水），请批评他并提醒少喝含糖饮料，建议换成水。"
            elif reminder_type == "phone":
                prompt_instruction = "用户在玩手机，请非常严厉地批评，要求立即放下手机回到工作状态。"
            elif reminder_type == "sitting":
                prompt_instruction = "用户已久坐超过30分钟，请提醒他站起来活动一下，以防久坐带来的健康问题。"
        elif should_encourage:
            if current_behavior == "work":
                prompt_instruction = "用户持续工作一段时间了，请赞扬他的专注和努力，给予积极鼓励。"
            elif current_behavior == "drinking_water":
                prompt_instruction = "用户在喝水，请表示赞同，鼓励多喝水保持健康。"
        else:
            # 如果没有特殊提示，使用一般性提示
            prompt_instruction = f"根据检测到的行为类型'{behavior_desc}'给出相应回应。如果是工作或喝水，给予鼓励；如果是吃东西、玩手机、喝饮料或睡觉，给予批评和提醒。"
        
        # 添加当前观察到聊天上下文，更新聊天上下文（给 AI 的历史对话）
        user_message = {"role": "user", "content": f"观察结果: {analysis_text}\n\n{prompt_instruction}"}
        self.chat_context.append(user_message)
        
        # 限制上下文长度，避免超出token限制
        if len(self.chat_context) > 20:  # 保留最近20条消息
            # 保留系统消息和最近的消息
            self.chat_context = [self.chat_context[0]] + self.chat_context[-19:]
        



        #调用 AI 模型生成回应 & 处理结果
        try:
            print(f"调用DeepSeek生成回应，消息历史长度: {len(self.chat_context)}")
            
            # 使用完整的聊天上下文
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=self.chat_context,  # 使用累积的对话历史
                stream=False
            )
            assistant_reply = response.choices[0].message.content
            print(f"DeepSeek回应: {assistant_reply}")
            
            # 将AI回应也添加到对话历史
            assistant_message = {"role": "assistant", "content": assistant_reply}
            self.chat_context.append(assistant_message)
            
            # 添加AI回应到聊天记录
            self.add_ai_message(assistant_reply)
            
            # 只有在需要提醒或鼓励时才播放语音
            if should_remind or should_encourage:
                self.audio_player.play_text(assistant_reply, priority=2)
        except Exception as e:
            error_msg = f"DeepSeek API error: {e}"
            print(error_msg)
            self.update_status(error_msg)

        # 流程：
        # 用 chat_context 作为输入调用 AI；
        # 获取 AI 的回复（比如 “别玩手机了，赶紧工作！”）；
        # 把回复存入上下文（方便后续对话参考）；
        # 显示到 UI，并在需要提醒 / 鼓励时，用语音播放（比如用户久坐时，语音提醒 “该站起来活动了”）。

        # 总结：这个函数干了什么？
        # 简单说，它是一个 “图像分析结果→行为跟踪→智能反馈” 的完整处理链：





    def check_timestamp(self):
        """Check if we need to display a new timestamp"""
        current_time = time.time()
        if current_time - self.last_timestamp >= self.timestamp_interval:
            self.add_timestamp()
            self.last_timestamp = current_time
        
        # Schedule the next check
        self.after(5000, self.check_timestamp)  # Check every 5 seconds
    
    def add_timestamp(self):
        """Add a timestamp to the chat UI"""
        # Get current time in the required format
        now = datetime.now()
        time_str = now.strftime("%m月%d日 %H:%M")
        
        # Create timestamp frame
        timestamp_frame = ctk.CTkFrame(self.chat_frame, fg_color=("#E0E0E0", "#3F3F3F"), corner_radius=15)
        timestamp_frame.grid(row=self.chat_row, column=0, pady=5)
        self.chat_row += 1
        
        # Add timestamp label - 使用自定义时间戳字体
        timestamp_label = ctk.CTkLabel(
            timestamp_frame, 
            text=time_str,
            font=self.timestamp_font,
            fg_color="transparent",
            padx=10,
            pady=2
        )
        timestamp_label.grid(row=0, column=0)
        
        # Scroll to bottom
        self.scroll_to_bottom()
    




    def add_ai_message(self, text, screenshot=None, is_placeholder=False, placeholder_id=None):
        #核心功能是在聊天界面中添加 AI 发送的消息，支持显示文本、截图，还能标记 “占位符消息”（临时显示的加载状态）
        """Add an AI message to the chat UI"""
        # 参数说明：
        # self：类实例本身（用于访问类的变量和其他方法）；
        # text：AI 消息的文本内容（比如 “你正在认真工作，继续加油！”）；
        # screenshot：可选的截图图片（比如摄像头捕捉的画面，用于配合文字展示）；
        # is_placeholder：是否为 “占位符消息”（临时显示，比如 “正在分析画面...”，后续会被替换）；
        # placeholder_id：占位符消息的唯一标识（用于后续找到这条消息并更新内容）。




        # Generate placeholder id if needed
        #生成占位符 ID（临时消息的唯一标识）
        #如果是占位符消息且没提供 ID，自动生成一个唯一 ID（比如 ai_0、ai_1），方便后续更新这条临时消息。
        if is_placeholder and not placeholder_id:
            placeholder_id = f"ai_{self.message_id}"
            self.message_id += 1
        
        print(f"添加AI消息: 长度={len(text)}, 有截图={screenshot is not None}, 是占位符={is_placeholder}, ID={placeholder_id}")
        
        # Create message frame
        #创建一个 “消息容器”（CTkFrame 组件），用于包裹 AI 的头像、名称、文本、图片等内容。
        message_frame = ctk.CTkFrame(self.chat_frame, fg_color=("#EAEAEA", "#2B2B2B"))
        message_frame.grid(row=self.chat_row, column=0, sticky="w", padx=5, pady=5)
        message_frame.grid_columnconfigure(1, weight=1)
        #重温细节：
            # fg_color=("#EAEAEA", "#2B2B2B")：设置背景色（浅色模式为浅灰，深色模式为深灰）；
            # grid(row=self.chat_row, ...)：通过 grid 布局放在聊天区域（self.chat_frame）的第 self.chat_row 行（确保消息按顺序显示）；
            # sticky="w"：消息左对齐（AI 消息通常靠左显示，用户消息靠右）；
            # grid_columnconfigure(1, weight=1)：第 1 列（文本 / 图片列）设置权重 1，确保内容能自适应窗口宽度。
                    
        # Store placeholder row if needed
        if is_placeholder and placeholder_id:
            self.placeholder_map[placeholder_id] = self.chat_row
            print(f"存储占位符 {placeholder_id} 在行 {self.chat_row}")
        #作用：如果是占位符消息，把它的 ID 和所在的行号存到 self.placeholder_map 字典中，后续需要更新时通过 ID 找到对应的行。
        

        self.chat_row += 1
        ## 行号+1，下次添加消息时会显示在新行

        # Add avatar
        avatar_label = ctk.CTkLabel(message_frame, image=self.ai_avatar, text="")
        avatar_label.grid(row=0, column=0, rowspan=2, padx=5, pady=5)
            # 作用：在消息容器的左侧显示 AI 的头像。
            # 细节：
            # image=self.ai_avatar：使用提前加载的 AI 头像图片（比如蓝色圆形图标）；
            # text=""：清空文字（只显示图片）；
            # grid(row=0, column=0, rowspan=2)：放在第 0 行、第 0 列，跨 2 行显示（和名称、文本对齐）；
            # padx=5, pady=5：设置边距，避免头像贴边。


        
        # Add name添加 AI 名称（如 “DeepSeek”）
        name_label = ctk.CTkLabel(message_frame, text="DeepSeek", font=("Arial", 12, "bold"), 
                                  anchor="w", fg_color="transparent")
        name_label.grid(row=0, column=1, sticky="w", padx=5, pady=(5, 0))
        


        # Add screenshot if provided
        #这部分是核心，负责在消息中显示截图：
        if screenshot is not None:
            try:
                # 创建图片容器（避免图片和文字挤在一起）
                img_frame = ctk.CTkFrame(message_frame, fg_color="transparent")
                img_frame.grid(row=1, column=1, sticky="w", padx=5, pady=5)
                
                # 检查图片是否有效（是否有copy方法，确保是PIL Image对象）
                if hasattr(screenshot, 'copy'):
                    # Resize the image for display
                    img_resized = screenshot.copy()# 复制原图（避免修改原图）
                    img_resized.thumbnail((200, 150))# 限制最大尺寸为200x150
                    
                    # 转换为CTk支持的图片格式（CTkImage）
                    ctk_img = ctk.CTkImage(
                        light_image=img_resized,  # 浅色模式图片
                        dark_image=img_resized,  # 深色模式图片（这里和浅色一样）
                        size=(200, 150)  # 显示尺寸
                    )
                    
                    # 创建图片标签并显示
                    img_label = ctk.CTkLabel(img_frame, image=ctk_img, text="")
                    img_label.grid(row=0, column=0, padx=2, pady=2)
                    
                    # 关键：保留图片引用，防止被Python垃圾回收机制删除
                    img_label.image = ctk_img
                    
                    print(f"成功添加图片: {img_resized.size}")
                else:
                    ## 图片无效（没有copy方法）
                    error_msg = "图像对象无copy属性"
                    print(error_msg)
                    error_label = ctk.CTkLabel(img_frame, text=f"[图像处理错误: {error_msg}]")
                    error_label.grid(row=0, column=0, padx=2, pady=2)
            except Exception as e:
                # 其他图像处理错误（比如图片损坏）
                print(f"图像处理错误: {e}")
                error_label = ctk.CTkLabel(message_frame, text=f"[图像处理错误: {str(e)}]")
                error_label.grid(row=1, column=1, padx=5, pady=5, sticky="w")
            
                # 作用：如果有截图，将图片处理后显示在消息中（头像右侧、名称下方），并在图片下方显示文本。
                # 关键细节：
                # 图片缩放：thumbnail((200, 150)) 确保图片不会太大，避免界面混乱；
                # 格式转换：CTkImage 是 customtkinter 专用的图片格式，必须转换才能显示；
                # 保留引用：img_label.image = ctk_img 非常重要！如果不保留，Python 会自动删除图片数据，界面上图片会消失；
                # 错误处理：如果图片无效或处理失败，显示错误提示（而不是崩溃）。

            text_label = ctk.CTkLabel(message_frame, text=text, wraplength=600, justify="left", 
                                     anchor="w", fg_color="transparent")
            text_label.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        else:
            #纯文本消息显示（无截图时）
            text_label = ctk.CTkLabel(message_frame, text=text, wraplength=600, justify="left", 
                                     anchor="w", fg_color="transparent")
            text_label.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        #调整占位符样式（区分临时消息）
        if is_placeholder:
            message_frame.configure(fg_color=("#F5F5F5", "#3B3B3B"))
            if screenshot is not None:
                text_label.configure(text_color=("#888888", "#AAAAAA"))
            else:
                text_label.configure(text_color=("#888888", "#AAAAAA"))
        
        #滚动到最底部（显示最新消息）
        self.after(100, self.scroll_to_bottom)# 100毫秒后调用滚动方法
        #作用：添加消息后，自动滚动聊天窗口到底部，确保用户能看到最新消息。
        
        # Return placeholder id if applicable
        return placeholder_id if is_placeholder else None
    





    def add_user_message(self, text, is_placeholder=False, replace_placeholder=None, placeholder_id=None):
        """Add a user message to the chat UI"""
        #作用是在聊天界面中添加用户发送的消息，功能和 add_ai_message 类似，但针对 “用户消息” 的 UI 样式（如对齐方式、颜色）做了专门设计
        #         作用：在聊天界面添加用户消息（和 add_ai_message 对应，分别处理用户和 AI 的消息）。
        # 参数说明（重点看和 add_ai_message 的区别）：
        # text：用户消息的文本内容（比如 “我刚才在喝水吗？”）；
        # is_placeholder：是否为临时占位符（比如 “正在录音...”）；
        # replace_placeholder：要替换的旧占位符 ID（比如之前显示 “正在输入”，现在用实际消息替换它）；
        # placeholder_id：当前消息的占位符 ID（如果是临时消息）。
                
        
        print(f"添加用户消息: '{text[:30]}...', 占位符={is_placeholder}, 替换ID={replace_placeholder}, 新ID={placeholder_id}")
        
        # 替换旧占位符（核心差异点）
        if replace_placeholder and replace_placeholder in self.placeholder_map:
            print(f"从映射中移除占位符: {replace_placeholder}")
            # In a full implementation, we would update the existing widget
            # But for simplicity, we just add a new message
            del self.placeholder_map[replace_placeholder]
            #用户按住录音键时，先调用 add_user_message 创建一个占位符 “正在录音...”（ID 为 user_5）；录音完成后，
            # 调用 add_user_message 并传入 replace_placeholder="user_5"，就会删除旧占位符的记录，用新消息替换它。
        
        #  生成用户占位符 ID
        if is_placeholder and not placeholder_id:
            placeholder_id = f"user_{self.message_id}"
            self.message_id += 1
            print(f"生成新占位符ID: {placeholder_id}")
        
        #  创建用户消息容器（Frame）
        message_frame = ctk.CTkFrame(self.chat_frame, fg_color=("#C7E9C0", "#2D3F2D"))
        message_frame.grid(row=self.chat_row, column=0, sticky="e", padx=5, pady=5)
        
        # S存储用户占位符位置
        if is_placeholder and placeholder_id:
            self.placeholder_map[placeholder_id] = self.chat_row
            print(f"存储占位符 {placeholder_id} 在行 {self.chat_row}")
            
        self.chat_row += 1
        
        # Add avatar
        avatar_label = ctk.CTkLabel(message_frame, image=self.user_avatar, text="")
        avatar_label.grid(row=0, column=1, rowspan=2, padx=5, pady=5)
        
        # Add name
        name_label = ctk.CTkLabel(message_frame, text="User", font=("Arial", 12, "bold"), 
                                  anchor="e", fg_color="transparent")
        name_label.grid(row=0, column=0, sticky="e", padx=5, pady=(5, 0))
        
        # Add text
        text_label = ctk.CTkLabel(message_frame, text=text, wraplength=600, justify="right", 
                                  anchor="e", fg_color="transparent")
        text_label.grid(row=1, column=0, sticky="e", padx=5, pady=5)
        
        # Mark as placeholder with different color if needed
        if is_placeholder:
            message_frame.configure(fg_color=("#DCF0D5", "#394639"))
            text_label.configure(text_color=("#888888", "#AAAAAA"))
        
        # Scroll to bottom
        self.after(100, self.scroll_to_bottom)
        
        # Return placeholder id if applicable
        return placeholder_id if is_placeholder else None
        #add_user_message 是专门为 “用户消息” 设计的 UI 渲染方法





    def scroll_to_bottom(self):
        """更可靠地滚动聊天视图到底部"""
        try:
            # 使用after方法确保在UI更新后执行滚动
            self.after(10, lambda: self._do_scroll_to_bottom())
        except Exception as e:
            print(f"Scroll error: {e}")



    def _do_scroll_to_bottom(self):
        """实际执行滚动的内部方法"""
        try:
            # 获取可滚动区域的画布
            canvas = self.chat_frame._parent_canvas
            
            # 获取画布的内容高度
            canvas.update_idletasks()  # 确保更新布局
            
            # 明确设置滚动区域底部位置
            canvas.yview_moveto(1.0)
            
            # 额外的方法确保滚动到底部
            canvas.update_idletasks()
            canvas.yview_scroll(1000000, "units")  # 大数字确保滚动到底部
        except Exception as e:
            print(f"Detailed scroll error: {e}")
            

    
    def update_preview(self, img):
        """This method is now deprecated but kept for compatibility"""
        pass
    
    def update_status(self, text):
        """Update the status message"""
        self.status_label.configure(text=text)
    
    def analyze_images(self, image_urls, screenshots, current_screenshot, placeholder_id=None):
        #核心功能是将图像发送给 Qwen-VL 视觉语言模型 API 进行分析，判断用户当前的行为（如工作、吃东西、玩手机等），
        # 并将分析结果传递给后续流程处理。它是连接 “图像采集” 和 “行为分析反馈” 的关键环节。
        #         参数说明：
        # self：类实例本身（访问类变量和方法）；
        # image_urls：图像的 URL 列表（已上传到 OSS 等存储，供 API 访问）；
        # screenshots：截图数据（可能用于后续 UI 显示）；
        # current_screenshot：当前截图（用于后续在 UI 中展示对应的分析结果）；
        # placeholder_id：UI 中对应的占位符 ID（后续用分析结果更新这个占位符）。
        """Send images to Qwen-VL for analysis"""
        #检查图像 URL 是否有效
        if not image_urls:
            print("没有图像URL可供分析")
            return
        
        #更新状态与打印调试信息
        self.update_status("正在分析图像...")
        print(f"分析图像: {len(image_urls)} URLs, 占位符ID: {placeholder_id}")
        
        #构建发送给 Qwen-VL 的消息
        messages = [{
            "role": "system",
            "content": [{"type": "text", "text": "详细观察这个人正在做什么。务必判断他属于以下哪种情况：1.认真专注工作, 2.吃东西, 3.用杯子喝水, 4.喝饮料, 5.玩手机, 6.睡觉, 7.其他。分析他的表情、姿势、手部动作和周围环境来作出判断。使用中文回答，并明确指出是哪种情况。"}]
        }]
        
        message_payload = {
            "role": "user",
            "content": [
                {"type": "video", "video": image_urls},
                {"type": "text", "text": "这个人正在做什么？请判断他是：1.认真专注工作, 2.吃东西, 3.用杯子喝水, 4.喝饮料, 5.玩手机, 6.睡觉, 7.其他。请详细描述你观察到的内容并明确指出判断结果。"}
            ]
        }
        messages.append(message_payload)
        
        #调用 Qwen-VL API 获取分析结果
        try:
            print("调用Qwen-VL API进行图像分析...")
            completion = qwen_client.chat.completions.create(
                model="qwen-vl-max",
                messages=messages,
            )
            analysis_text = completion.choices[0].message.content
            print(f"图像分析完成，分析长度: {len(analysis_text)} 字符")
            
           # 从分析文本中提取行为编号和描述（调用之前学过的extract_behavior_type函数）
            behavior_num, behavior_desc = extract_behavior_type(analysis_text)
            
            # 记录行为到日志
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"{timestamp}-{behavior_num}-{analysis_text}"
            logging.info(log_message)
            print(f"行为记录已保存到日志: {behavior_num}-{behavior_desc}")
            
            ## 将分析结果添加到消息队列，等待后续处理
            # Add to message queue for processing with appropriate priority
            # Priority 2 for normal image analysis (voice input would be priority 1)
            print("添加分析结果到消息队列")
            self.message_queue.put((
                2, # 优先级（数字越小优先级越高，图像分析为2，语音输入为1）
                self.message_id,  # message id for sequence
                {
                    "type": "image_analysis",
                    "content": analysis_text,
                    "urls": image_urls,
                    "screenshots": [current_screenshot] if current_screenshot else [],
                    "placeholder_id": placeholder_id
                }
            ))
            #将分析结果封装成消息，放入优先级队列（message_queue），由后台线程（process_message_queue）处理。
            self.message_id += 1
            
        except Exception as e:
            error_msg = f"Qwen-VL API error: {e}"
            print(error_msg)
            self.update_status(error_msg)
    
    def transcribe_audio(self, audio_file, priority=False, placeholder_id=None):
        #核心功能是将录制的音频文件通过 ASR（自动语音识别）模型（这里用的是 SenseVoice）转录成文本，
        # 并将转录结果放入消息队列供后续处理（比如生成 AI 回应）

        #想象用户对着麦克风说话，系统录制了音频（比如 “我刚才在喝水吗？”），
        # 这个方法就负责把这段音频 “翻译” 成文字，让系统知道用户说了什么，之后才能进一步分析问题并回答。

        #         参数说明：
        # self：类实例本身（访问类变量和方法）；
        # audio_file：音频文件路径（需要转录的音频，比如 speech_123.wav）；
        # priority：是否为高优先级（True 表示语音输入需要优先处理，比如用户主动说话）；
        # placeholder_id：对应的 UI 占位符 ID（后续用转录结果更新这个占位符）。
        """Transcribe recorded audio using SenseVoice"""
        self.update_status("正在转录语音...")
        print(f"转录音频: {audio_file}, 优先级: {priority}, 占位ID: {placeholder_id}")
        
        try:
            # 前置检查：音频文件是否有效
            if not os.path.exists(audio_file):
                error_msg = f"音频文件不存在: {audio_file}"
                print(error_msg)
                self.update_status(error_msg)
                return
            
            # 检查文件大小（避免空文件）
            file_size = os.path.getsize(audio_file)
            print(f"音频文件大小: {file_size} 字节")
            if file_size == 0:
                error_msg = "音频文件为空"
                print(error_msg)
                self.update_status(error_msg)
                return
            
            # 调用 ASR 模型进行转录
            print("调用ASR模型转录...")
            res = asr_model.generate(
                input=audio_file,
                cache={},
                language="auto",
                use_itn=False,
                ban_emo_unk=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )
            
            print(f"ASR结果: {res}")
            
            #处理转录结果：提取有效文本
            if len(res) > 0 and "text" in res[0]:
                text = res[0]["text"]
                extracted_text = extract_language_emotion_content(text)
                print(f"提取的文本内容: {extracted_text}")
                
                # 新增：检查提取的文本是否为空或太短（可能是噪音）
                if not extracted_text or len(extracted_text.strip()) < 2:
                    print(f"检测到空语音或噪音: '{extracted_text}'，忽略处理")
                    self.update_status("检测到噪音，忽略")
                    return

                # 关键细节：
                # extract_language_emotion_content 函数：去掉原始文本中的标记（如 |zh|neutral|>），只保留纯文本（比如从 |zh|neutral|> 我刚才在喝水吗 提取出 我刚才在喝水吗）；
                # 长度检查：如果文本为空或太短（比如只有 “啊”“嗯”），视为噪音，不继续处理，避免无效交互。

                # Add to message queue with high priority if requested
                priority_level = 1 if priority else 2
                
                print(f"添加语音输入到消息队列，优先级: {priority_level}")
                #将转录结果放入消息队列’#
                #作用：将有效的转录文本封装成消息，放入优先级队列，由后台线程处理（后续会调用 process_voice_input 生成 AI 回应）。
                self.message_queue.put((
                    priority_level,  # priority (lower number = higher priority)
                    self.message_id,  # message id for sequence
                    {
                        "type": "voice_input",
                        "content": extracted_text,
                        "placeholder_id": placeholder_id
                    }
                ))
                self.message_id += 1
                
                # 高优先级语音中断当前播放
                #当用户主动说话（高优先级）时，立即停止系统正在播放的语音（比如之前的提醒），确保用户能快速得到回应，提升交互体验。
                if priority:
                    print("语音输入优先，跳过当前语音播放")
                    self.audio_player.skip_current()
            else:
                error_msg = "未检测到语音或转录失败"
                print(error_msg)
                self.update_status(error_msg)
                
        except Exception as e:
            error_msg = f"转录错误: {e}"
            print(error_msg)
            self.update_status(error_msg)
    
    def start_voice_recording(self):
        """Start recording voice when 'r' key is pressed"""
        # This is retained for backwards compatibility, but the continuous
        # voice detection has replaced this functionality
        self.update_status("使用自动语音检测 - 直接说话即可")
    

    def stop_voice_recording(self):
        """Stop recording voice when 's' key is pressed"""
        # This is retained for backwards compatibility, but the continuous
        # voice detection has replaced this functionality
        pass
    

    def skip_audio(self):
        """Skip currently playing audio and toggle analysis pause when spacebar is pressed"""
        self.audio_player.skip_current()
        self.webcam_handler.toggle_pause()
        
        # Show/hide camera window
        if self.webcam_handler.camera_window and self.webcam_handler.camera_window.is_closed:
            self.webcam_handler.create_camera_window()
        elif self.webcam_handler.camera_window and not self.webcam_handler.camera_window.is_closed:
            self.webcam_handler.camera_window.on_closing()

# ---------------- Main Function ----------------
def main():
    # Set appearance mode and default theme
    ctk.set_appearance_mode("System")  # "System", "Dark" or "Light"
    ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"
    
    app = MultimediaAssistantApp()
    app.protocol("WM_DELETE_WINDOW", lambda: quit_app(app))
    app.mainloop()

def quit_app(app):
    """Clean shutdown of the application"""
    # Stop all threads
    if hasattr(app, 'webcam_handler'):
        app.webcam_handler.stop()
    
    if hasattr(app, 'voice_detector'):
        app.voice_detector.stop_monitoring()
    
    if hasattr(app, 'processing_running'):
        app.processing_running = False
        
    if hasattr(app, 'audio_player'):
        app.audio_player.stop()
        
    # Clean up keyboard handlers
    keyboard.unhook_all()
    
    # Clean up temporary files
    try:
        for file in os.listdir():
            if file.startswith("output") and (file.endswith(".mp3") or file.endswith(".wav")):
                os.remove(file)
                print(f"删除临时文件: {file}")
    except Exception as e:
        print(f"清理临时文件时出错: {e}")
    
    # Close the app
    app.destroy()

if __name__ == "__main__":
    main()