import os
import cv2
import time
import io
import threading
import queue
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
import oss2
from datetime import datetime, timedelta
import re
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from openai import OpenAI
import matplotlib.font_manager as fm

# 日志配置
LOG_FILE = "behavior_logg.txt"  # 定义日志文件名
logging.basicConfig(
    filename=LOG_FILE,         # 指定日志输出到文件
    level=logging.INFO,        # 设置日志级别为INFO(只记录INFO及以上级别的日志)
    format='%(asctime)s - %(message)s',  # 定义日志格式: 时间 - 消息
    datefmt='%Y-%m-%d %H:%M:%S'  # 定义时间格式: 年-月-日 时:分:秒
)

# 设置中文字体支持
# 尝试加载系统默认中文字体
try:
    # 尝试常见中文字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'NSimSun', 'FangSong', 'KaiTi']
    chinese_font = None
    
    for font_name in chinese_fonts:
        try:
            # 检查字体是否可用
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if os.path.exists(font_path):
                chinese_font = font_name
                break
        except:
            continue
    
    if chinese_font:
        plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans']
    else:
        # 如果没有找到中文字体，使用默认字体并记录警告
        print("警告：未找到中文字体，某些文本可能显示不正确")
        
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
except Exception as e:
    print(f"设置中文字体时出错: {e}")

# ---------------- API客户端初始化 ----------------
# Qwen-VL客户端
qwen_client = OpenAI(
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL
)

# ---------------- 工具函数 ----------------
def extract_behavior_type(analysis_text):
    """从AI分析文本中提取行为类型编号"""
    # 尝试在文本中查找行为类型编号(1-7)
    pattern = r'(\d+)\s*[.、:]?\s*(认真专注工作|吃东西|用杯子喝水|喝饮料|玩手机|睡觉|其他)'
    match = re.search(pattern, analysis_text)
    
    if match:
        behavior_num = match.group(1)
        behavior_desc = match.group(2)
        return behavior_num, behavior_desc
    
    # 如果第一种模式失败，尝试替代模式
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
    
    return "0", "未识别"  # 如果没有匹配项，返回默认值

# ---------------- 摄像头显示窗口 ----------------
class CameraWindow(ctk.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # “self 是一个对象，self.xxx 就是我人为给它赋值的属性。”
        # *args 是接收任意数量的位置参数
        # **kwargs 是接收任意数量的关键字参数

        self.title("Camera Feed")  # 设置窗口标题
        self.geometry("640x480")   # 固定窗口尺寸为640x480像素
        self.protocol("WM_DELETE_WINDOW", self.on_closing)  # 绑定窗口关闭事件:protocol（正式的国际性）条约，公约，议定书
        # WM_DELETE_WINDOW 不能改变，这是捕获命令
        self.configure(fg_color="#1a1a1a")  # 深色背景 configure:配置
        
        # 创建摄像头显示框架
        self.camera_frame = ctk.CTkFrame(self, fg_color="#1a1a1a")
        self.camera_frame.pack(fill="both", expand=True, padx=10, pady=10)
        #pack参数说明
        #编写一个程序的界面，就是要把各个组件，以适当大小，定位到界面的某个位置。
        #https://blog.csdn.net/hyf64/article/details/121220643
        # 🔍 作用：
        # 将刚才创建的 camera_frame 添加到主窗口上，并设置其布局方式。
        # fill="both"：在横向和纵向都填满父容器。
        # expand=True：允许组件在窗口大小变化时自动扩大。
        # padx=10, pady=10：左右和上下的内边距为10像素。

        # 创建摄像头图像标签
        self.camera_label = ctk.CTkLabel(self.camera_frame, text="正在启动摄像头...", text_color="white")
        self.camera_label.pack(fill="both", expand=True)
        # 🔍 作用：
        # 在 camera_frame 中创建一个 文本标签（Label），初始显示提示“正在启动摄像头...”。
        # ctk.CTkLabel：CustomTkinter提供的标签控件。
        
        # 图像保存器
        self.current_image = None
        
        # 标记窗口是否关闭
        self.is_closed = False
    
    def update_frame(self, img):
        """更新摄像头帧，“每一帧刷新一次画面”的函数。"""
        if self.is_closed:
            return
            
        try:
            if img:
                # 调整图像大小以适应窗口
                img_resized = img.copy()
                img_resized.thumbnail((640, 480))
                # 复制一份图像，然后用 thumbnail 方法
                # 压缩图像大小为最多 640×480，以适配显示区域。不会拉伸，只是按比例缩小。
                # 转换为CTkImage
                ctk_img = ctk.CTkImage(light_image=img_resized, dark_image=img_resized, size=(640, 480))
                # 将 PIL 图像转换为 CustomTkinter 的 CTkImage 类型
                # 更新标签
                self.camera_label.configure(image=ctk_img, text="")
                # 把摄像头图像设置到之前的 camera_label 上。
                # 原本显示的是“正在启动摄像头...”，现在改为显示图像，并把文字清空了。
                
                # 保存引用以防止垃圾回收
                self.current_image = ctk_img
        except Exception as e:
            print(f"更新摄像头帧出错: {e}")
    
    def on_closing(self):
        """处理窗口关闭事件"""
        self.is_closed = True
        self.withdraw()  # 隐藏而不是销毁，以便重新打开

# ---------------- 行为可视化类 ----------------
class BehaviorVisualizer:
    """处理检测到的行为的可视化"""
    
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.behavior_map = {
            "1": "专注工作",
            "2": "吃东西",
            "3": "喝水",
            "4": "喝饮料",
            "5": "玩手机",
            "6": "睡觉",
            "7": "其他"
        }
        
        # 不同行为的颜色（确保两个图表中的颜色一致）
        self.behavior_colors = {
            "1": "#4CAF50",  # 绿色表示工作
            "2": "#FFC107",  # 琥珀色表示吃东西
            "3": "#2196F3",  # 蓝色表示喝水
            "4": "#9C27B0",  # 紫色表示喝饮料
            "5": "#F44336",  # 红色表示玩手机
            "6": "#607D8B",  # 蓝灰色表示睡觉
            "7": "#795548"   # 棕色表示其他
        }
        
        # 数据存储
        self.behavior_history = []  # (时间戳, 行为编号) 元组列表
        self.behavior_counts = {key: 0 for key in self.behavior_map}
        
        # 图表更新频率
        self.update_interval = 2  # 秒
        
        # 设置图表
        self.setup_charts()
        
        # 启动更新线程
        self.running = True
        self.update_thread = threading.Thread(target=self._update_charts_thread)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def setup_charts(self):
        """创建并设置折线图和饼图"""
        # 创建图表主框架
        self.charts_frame = ctk.CTkFrame(self.parent_frame, fg_color="#1a1a1a")
        self.charts_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 创建左侧面板放置折线图（占据大部分空间）
        self.line_chart_frame = ctk.CTkFrame(self.charts_frame, fg_color="#1a1a1a")
        self.line_chart_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        # 创建右侧面板放置饼图
        self.right_panel = ctk.CTkFrame(self.charts_frame, fg_color="#1a1a1a")
        self.right_panel.pack(side="right", fill="both", expand=False, padx=5, pady=5, ipadx=10)
        
        # 创建饼图框架
        self.pie_chart_frame = ctk.CTkFrame(self.right_panel, fg_color="#1a1a1a")
        self.pie_chart_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 设置折线图
        self.setup_line_chart()
        
        # 设置饼图
        self.setup_pie_chart()
        
        # 添加刷新按钮
        self.refresh_button = ctk.CTkButton(
            self.right_panel, 
            text="刷新图表", 
            command=self.refresh_charts,
            fg_color="#333333",
            text_color="white",
            hover_color="#555555"
        )
        self.refresh_button.pack(pady=10, padx=10)
        
        # 初始化空的统计标签字典（仍需保留以避免其他方法的引用错误）
        self.stat_labels = {}
        self.color_frames = {}
    
    def setup_line_chart(self):
        """设置行为跟踪随时间变化的折线图"""
        # 创建matplotlib图形和轴 - 增加宽度以充分利用900px宽度
        self.line_fig = Figure(figsize=(7, 3.8), dpi=100)
        self.line_fig.patch.set_facecolor('#1a1a1a')  # 设置图形背景为黑色
        self.line_ax = self.line_fig.add_subplot(111)
        self.line_ax.set_facecolor('#1a1a1a')  # 设置绘图区背景为黑色
        
        # 设置标题和标签颜色为白色
        self.line_ax.set_title("行为随时间变化", color='white')
        self.line_ax.set_xlabel("时间", color='white')
        self.line_ax.set_ylabel("行为", color='white')
        
        # 设置刻度标签为白色
        self.line_ax.tick_params(axis='x', colors='white')
        self.line_ax.tick_params(axis='y', colors='white')
        
        # 设置边框颜色为白色
        for spine in self.line_ax.spines.values():
            spine.set_edgecolor('white')
        
        # 设置y轴显示行为类型
        self.line_ax.set_yticks(list(range(1, 8)))
        self.line_ax.set_yticklabels([self.behavior_map[str(i)] for i in range(1, 8)])
        
        # 添加网格
        self.line_ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        # 嵌入到Tkinter
        self.line_canvas = FigureCanvasTkAgg(self.line_fig, master=self.line_chart_frame)
        self.line_canvas.draw()
        self.line_canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def setup_pie_chart(self):
        """设置行为分布饼图"""
        # 创建matplotlib图形和轴 - 设置更大的底部空间给图例
        self.pie_fig = Figure(figsize=(3.5, 3.8), dpi=100)
        self.pie_fig.patch.set_facecolor('#1a1a1a')  # 设置图形背景为黑色
        self.pie_ax = self.pie_fig.add_subplot(111)
        self.pie_ax.set_facecolor('#1a1a1a')  # 设置绘图区背景为黑色
        # 调整子图位置，腾出底部空间给图例
        self.pie_fig.subplots_adjust(bottom=0.2)
        
        # 设置标题颜色为白色
        self.pie_ax.set_title("行为分布", color='white')
        
        # 初始时不显示任何数据，只显示一个空的圆
        self.pie_ax.text(0, 0, "等待数据...", ha='center', va='center', color='white', fontsize=12)
        self.pie_ax.set_aspect('equal')
        self.pie_ax.axis('off')  # 隐藏坐标轴
        
        # 嵌入到Tkinter
        self.pie_canvas = FigureCanvasTkAgg(self.pie_fig, master=self.pie_chart_frame)
        self.pie_canvas.draw()
        self.pie_canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def add_behavior_data(self, timestamp, behavior_num, behavior_desc):
        """向可视化添加新的行为数据点"""
        try:
            # 添加到历史记录
            self.behavior_history.append((timestamp, behavior_num))
            
            # 更新计数
            self.behavior_counts[behavior_num] = self.behavior_counts.get(behavior_num, 0) + 1
            
            # 限制历史记录长度以提高性能（保留最近100个条目）
            if len(self.behavior_history) > 100:
                self.behavior_history = self.behavior_history[-100:]
                
            print(f"添加行为数据: {behavior_num} - {behavior_desc}")
            
            # 不立即更新图表，更新线程会处理此操作
        except Exception as e:
            print(f"添加行为数据时出错: {e}")
    
    def _update_charts_thread(self):
        """定期更新图表的线程"""
        while self.running:
            try:
                # 更新折线图
                self.update_line_chart()
                
                # 更新饼图
                self.update_pie_chart()
                
                # 更新统计信息
                self.update_statistics()
            except Exception as e:
                print(f"更新图表时出错: {e}")
            
            # 等待下次更新
            time.sleep(self.update_interval)
    
    def update_line_chart(self):
        """用最新数据更新折线图"""
        try:
            self.line_ax.clear()
            
            # 设置背景颜色
            self.line_ax.set_facecolor('#1a1a1a')
            
            # 设置文本颜色为白色
            self.line_ax.set_title("行为随时间变化", color='white')
            self.line_ax.set_xlabel("时间", color='white')
            self.line_ax.set_ylabel("行为", color='white')
            self.line_ax.tick_params(axis='x', colors='white')
            self.line_ax.tick_params(axis='y', colors='white')
            
            # 设置边框颜色为白色
            for spine in self.line_ax.spines.values():
                spine.set_edgecolor('white')
            
            if not self.behavior_history:
                # 尚无数据，显示带有正确标签的空图表
                self.line_ax.set_yticks(list(range(1, 8)))
                self.line_ax.set_yticklabels([self.behavior_map[str(i)] for i in range(1, 8)])
                self.line_ax.grid(True, linestyle='--', alpha=0.3, color='gray')
                self.line_canvas.draw()
                return
            
            # 提取数据
            times, behaviors = zip(*self.behavior_history)
            
            # 将行为编号转换为整数以便绘图
            behavior_ints = [int(b) for b in behaviors]
            
            # 为每种行为创建散点图和线
            for i in range(1, 8):
                # 筛选此行为的数据
                indices = [j for j, b in enumerate(behavior_ints) if b == i]
                if indices:
                    behavior_times = [times[j] for j in indices]
                    behavior_vals = [behavior_ints[j] for j in indices]
                    
                    # 用正确的颜色绘制散点
                    self.line_ax.scatter(
                        behavior_times, 
                        behavior_vals, 
                        color=self.behavior_colors[str(i)],
                        s=50,  # 点的大小
                        label=self.behavior_map[str(i)]
                    )
            
            # 绘制连接相邻点的线
            self.line_ax.plot(times, behavior_ints, 'k-', alpha=0.3, color='white')
            
            # 将x轴格式化为时间
            self.line_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            
            # 设置时间范围，最多显示1小时的数据，如果数据较少则显示较少时间
            now = datetime.now()
            min_time = now - timedelta(hours=1)
            if times and times[0] < min_time:
                self.line_ax.set_xlim(min_time, now)
            elif times:
                self.line_ax.set_xlim(times[0], now)
            
            # 设置y轴
            self.line_ax.set_yticks(list(range(1, 8)))
            self.line_ax.set_yticklabels([self.behavior_map[str(i)] for i in range(1, 8)])
            self.line_ax.set_ylim(0.5, 7.5)  # 添加一些填充
            
            # 添加网格
            self.line_ax.grid(True, linestyle='--', alpha=0.3, color='gray')
            
            # 更新画布
            self.line_fig.tight_layout()
            self.line_canvas.draw()
            
        except Exception as e:
            print(f"更新折线图时出错: {e}")
    
    def update_pie_chart(self):
        """用最新分布更新饼图"""
        try:
            self.pie_ax.clear()
            
            # 设置背景颜色
            self.pie_ax.set_facecolor('#1a1a1a')
            
            # 设置标题颜色为白色
            self.pie_ax.set_title("行为分布", color='white')
            
            # 获取当前计数
            sizes = [self.behavior_counts.get(str(i), 0) for i in range(1, 8)]
            labels = list(self.behavior_map.values())
            colors = [self.behavior_colors[str(i)] for i in range(1, 8)]
            
            # 检查是否有数据
            if sum(sizes) == 0:
                # 没有数据，显示等待消息
                self.pie_ax.text(0, 0, "等待数据...", ha='center', va='center', color='white', fontsize=12)
                self.pie_ax.set_aspect('equal')
                self.pie_ax.axis('off')  # 隐藏坐标轴
            else:
                # 有数据，显示饼图
                wedges, texts, autotexts = self.pie_ax.pie(
                    sizes,
                    labels=None,
                    colors=colors,
                    autopct='%1.1f%%',
                    startangle=90,
                    textprops={'color': 'white'}
                )
                
                # 添加图例到饼图下方而不是右侧
                legend = self.pie_ax.legend(wedges, labels, title="行为类型", 
                              loc="upper center", bbox_to_anchor=(0.5, -0.1),
                              frameon=False, labelcolor='white', fontsize='small', ncol=2)
                # 单独设置标题颜色
                plt.setp(legend.get_title(), color='white')
            
            # 更新画布
            self.pie_canvas.draw()
            
        except Exception as e:
            print(f"更新饼图时出错: {e}")
    
    def update_statistics(self):
        """用最新数据更新统计标签"""
        # 由于我们已删除统计标签区域，此方法保留但不执行任何操作
        pass
    
    def refresh_charts(self):
        """手动刷新所有图表"""
        self.update_line_chart()
        self.update_pie_chart()
        self.update_statistics()
    
    def stop(self):
        """停止更新线程"""
        self.running = False
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)

# ---------------- 摄像头处理类 ----------------
class WebcamHandler:
    def __init__(self, app):
        self.app = app
        #self.app 的真正用途是？？？？？？这里是极其绕的！！！
        #答：
        # 主窗口类：
        # class BehaviorVisualizationApp(ctk.CTk)://后面定义的
        #     def __init__(self):
        #         super().__init__()
        #         self.webcam_handler = WebcamHandler(self)
        # WebcamHandler 类：
        # class WebcamHandler:
        #     def __init__(self, app):
        #         self.app = app

        #！！！self.webcam_handler = WebcamHandler(self)
        #这个 self 是 BehaviorVisualizationApp 的实例对象。
        #你传进去的这个 self（主窗口）被接收为参数 app；
        # 然后保存在 self.app 中；
        # 也就是说，从此以后 WebcamHandler 就可以通过 self.app.xxx 访问和操作主窗口的
        self.running = False 
        self.paused = False  # 标记分析是否暂停
        self.processing = False  # 标记分析是否正在进行
        self.cap = None  # OpenCV摄像头对象
        self.webcam_thread = None  # 线程对象
        self.last_webcam_image = None  # 存储最近的摄像头图像
        self.debug = True  # 设置为True启用调试输出
        
        # 顺序处理控制
        self.analysis_running = False
        
        # 摄像头窗口
        self.camera_window = None
    
    def start(self):
        """启动摄像头捕获进程"""
        if not self.running:
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    self.app.update_status("无法打开摄像头")
                    return False
                
                self.running = True
                #表示捕获摄像头成功！要修改参数
                
                # 创建摄像头窗口，下面定义了这个函数！
                self.create_camera_window()
                
                # 启动处理线程！重点！
                self.webcam_thread = threading.Thread(target=self._process_webcam)
                # 这里的threading.Thread()创建了一个新的线程对象。 
                # 英文代词thread：线程；多线程
                # target=self._process_webcam表示这个线程启动后会执行 self._process_webcam 这个方法。
                # 也就是说，你把_process_webcam这个方法扔给了“新帮手”，让它在后台跑。


                self.webcam_thread.daemon = True
                #daemon：（古希腊神话中的）半神半人的精灵->守护进程；守护程序。
                # 让这个线程变成守护线程，
                # 守护线程的含义是：当主程序关闭时，这个线程会自动结束，不会阻塞程序退出。
                # 这样写一般是为了避免程序关闭时这个后台线程还在跑导致卡死。
                self.webcam_thread.start()
                # 这句才是真正启动线程，
                # 线程开始执行你给定的_process_webcam函数。
                # 一旦调用start()，后台线程就开始独立执行了。
                
                # 启动分析（重要 - 这将启动第一次捕获）
                self.analysis_running = True
                
                # 短暂延迟后启动首次分析
                self.app.after(2000, self.trigger_next_capture)
                
                return True
            except Exception as e:
                self.app.update_status(f"启动摄像头时出错: {e}")
                return False
        return False
    
    def create_camera_window(self):
        """创建显示摄像头画面的窗口"""
        if not self.camera_window or self.camera_window.is_closed:
            self.camera_window = CameraWindow(self.app)
            self.camera_window.title("摄像头画面")
            # 将窗口定位在主窗口下方：设置摄像头窗口的位置和大小
            main_x = self.app.winfo_x()
            main_y = self.app.winfo_y()
            main_height = self.app.winfo_height()

            self.camera_window.geometry(f"640x480+{main_x}+{main_y + main_height + 10}")
            # 这句干了三件事：
            # 设置窗口大小为 640x480
            # 设置窗口左上角坐标为 (main_x, main_y + main_height + 10)，也就是主窗口正下方往下偏移10像素
            # "geometry" 是 tkinter 里定位窗口的格式："宽x高+x坐标+y坐标"
    
    def stop(self):
        # 这是当用户关闭摄像头时要调用的方法，用来：
        # 停止后台线程、
        # 释放资源（摄像头）、
        # 销毁窗口，清理状态。
        """停止摄像头捕获进程"""
        self.running = False
        self.analysis_running = False

        # self.running = False → 停止摄像头采集主循环 _process_webcam。
        # self.analysis_running = False → 不再继续触发分析流程。
        
        if self.cap:
            self.cap.release()
        # self.cap 是通过 cv2.VideoCapture(0) 打开的摄像头对象。
        # .release() 是 OpenCV 提供的关闭摄像头设备的方法。
        
        # 关闭摄像头窗口
        if self.camera_window:
            self.camera_window.destroy()
            self.camera_window = None
        # self.camera_window 是你之前打开的视频画面窗口。
        # .destroy() 是销毁窗口的意思，相当于把那个小窗口关掉。
        # self.camera_window = None 是清空变量，避免下次误以为窗口还存在。

    
    def _process_webcam(self):
        """主摄像头处理循环 - 仅保留最近的帧"""
        #它是一个**“后台线程执行的主循环”**，不断从摄像头采集图像、处理并更新到界面。
        last_ui_update_time = 0
        ui_update_interval = 0.05  # 以20fps更新UI
        #UI每 0.05秒更新一次图像 → 相当于最多20fps（1秒最多更新20次）
        while self.running:
            try:
                ret, frame = self.cap.read()
                #ret：是否成功（布尔值）
                #frame：读取到的图像数据（OpenCV格式）

                if not ret:
                    self.app.update_status("无法捕获画面")
                    time.sleep(0.1)
                    continue
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # OpenCV 默认使用 BGR 色彩顺序，而我们在 Tkinter 中要用的是 PIL 的 RGB 格式
                img = Image.fromarray(frame_rgb)
                # （简而言之就是图像载体转化）把 OpenCV 图像 → PIL 图像，后续 CustomTkinter 的窗口可以显示它
                
                # 存储最近的图像
                self.last_webcam_image = img
                
                # 用当前帧更新摄像头窗口
                current_time = time.time()
                if self.camera_window and not self.camera_window.is_closed and current_time - last_ui_update_time >= ui_update_interval:
                    self.camera_window.update_frame(img)
                    last_ui_update_time = current_time
                # self.camera_window：摄像头窗口对象是否存在
                # not self.camera_window.is_closed：窗口没被手动关闭
                # current_time - last_ui_update_time >= 0.05：距离上次更新已经超过 50ms

                time.sleep(0.03)  # ~30fps捕获
            except Exception as e:
                error_msg = f"摄像头错误: {e}"
                print(error_msg)
                self.app.update_status(error_msg)
                #非常重要！他不光会打印你的错误信息而且还会用这个函数显示出来！
                time.sleep(1)  # 暂停后重试

            #总结:_process_webcam(self):
            # 这段 _process_webcam() 方法是摄像头采集线程的核心循环，只要 self.running 为 True，线程就会不断执行：
            # 使用 OpenCV 的 read() 方法读取一帧摄像头画面；
            # 将图像从 BGR 转为 RGB 并转为 PIL 格式；
            # 缓存为 self.last_webcam_image，供后续分析或截图使用；
            # 每隔 0.05 秒更新一次摄像头窗口中的图像画面（≈ 20fps）；
            # 每轮之间 sleep(0.03) 控制采集频率在 30fps 左右，节省资源；
            # 若发生异常，捕获错误、提示用户，并在 1 秒后自动重试。

    
    def trigger_next_capture(self):
        """触发下一次捕获和分析循环"""
        if self.running and self.analysis_running and not self.paused and not self.processing:
            print(f"触发新一轮图像分析 {time.strftime('%H:%M:%S')}")
            #别太在意！这是一个函数调用%H：小时（00–23）%M：分钟（00–59）%S：秒（00–59）
            self.capture_and_analyze()

            # ✅ 判断条件解释：
            # 条件	含义
            # self.running	摄像头线程在运行（摄像头已开启）
            # self.analysis_running	分析流程被激活（比如一开始设为 True）
            # not self.paused	当前没有暂停分析（比如按了暂停键）
            # not self.processing	当前没有正在分析的任务（避免并发分析）
            # 🧠 所以只有“摄像头开启 + 分析已启动 + 没暂停 + 没在处理中”时，才会触发下一轮分析！



    
    def capture_and_analyze(self):
        """捕获截图并发送进行分析"""
        if self.processing or self.paused:
            return
        
        try:
            self.processing = True
            self.app.update_status("捕捉图像中...")
            
            # 获取分析用的截图和当前显示用的截图
            screenshots, current_screenshot = self._capture_screenshots()
            # 如果到时候接入情感分析接口，可能就需要用到这两个参数！！！🖼️函数就在下面定义
            # _capture_screenshots() 是你自定义的函数，会从摄像头里捕获几张连续帧（用于分析），再单独捕获一帧用于显示。
            # 返回两个结果：
            # screenshots：多张分析用图
            # current_screenshot：当前展示用图

            
            # 在另一个线程中处理分析以保持UI响应，多线程处理分析任务
            analysis_thread = threading.Thread(
                target=self._analyze_screenshots, 
                args=(screenshots, current_screenshot)
            )
            analysis_thread.daemon = True
            analysis_thread.start()
                
        except Exception as e:
            error_msg = f"捕获/分析出错: {e}"
            print(error_msg)
            self.app.update_status(error_msg)
            self.processing = False
            # 延迟后重试
            self.app.after(2000, self.trigger_next_capture)


        # 上面这两个函数是一个“定时触发 → 安全判断 → 后台执行 → 自动继续”的行为分析循环核心，
        # 通过 after() 定时触发，threading.Thread 后台执行，保证主界面不卡顿，且分析任务不重复、不冲突。
        #你现在掌握了线程 + 定时器 + 异步分析 + UI联动四件套💪
        #总结：获取到了对应的截图，然后使用threading.Thread()函数开始调用target函数（下面），进行分析
        #->capture_and_analyze


    #开始正式分析！
    def _analyze_screenshots(self, screenshots, current_screenshot):
        """分析截图并更新UI"""
        try:
            self.app.update_status("正在分析图像...")
            
            # 将截图上传到OSS
            screenshot_urls = self._upload_screenshots(screenshots)
            #调用你自己定义的 _upload_screenshots() 函数（下面👇），把图像上传到阿里云 OSS
            #返回每张图像的访问链接（列表）

            if screenshot_urls:
                print(f"已上传 {len(screenshot_urls)} 张图片，开始分析")
                
                # 发送进行分析并等待结果（阻塞）
                analysis_text = self._get_image_analysis(screenshot_urls)
                #调用 _get_image_analysis() 发送图片 URL 给大模型（Qwen-VL）模型+返回分析结果（是字符串文本）
                
                if analysis_text:
                    print(f"分析完成")
                    
                    # 从分析文本中提取行为类型
                    behavior_num, behavior_desc = extract_behavior_type(analysis_text)
                    #这个函数是你自定义的行为标签提取器！
                    #VL：“这个人正在玩手机，低头看着屏幕，手里拿着一部智能手机。”
                    #behavior_num = 5；behavior_desc = 玩手机


                    # 记录行为到日志
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_message = f"{timestamp}-{behavior_num}-{analysis_text}"
                    #生成一条日志：时间 + 行为编号 + 模型原始分析结果
                    logging.info(log_message)
                    #调用 logging.info() 把它写入日志文件（或者控制台）
                    print(f"行为记录已保存到日志: {behavior_num}-{behavior_desc}")
                    
                    # 发送到行为可视化器更新图表
                    self.app.add_behavior_data(datetime.now(), behavior_num, behavior_desc, analysis_text)
                    #把数据（时间、编号、文字）发给前端 UI，比如可视化图表/曲线/列表更新
                    #实现行为分析结果的“实时可视化展示”


                    self.app.update_status(f"检测到行为: {behavior_desc}")
                else:
                    print("图像分析返回空结果")
            else:
                print("未能上传截图，无法进行分析")
        except Exception as e:
            error_msg = f"分析截图时出错: {e}"
            print(error_msg)
            self.app.update_status(error_msg)
        finally:#✅ 最后的 finally：无论成功或失败都做的事
            # 重要：标记为未处理并触发下一次捕获
            self.processing = False
            # 下次捕获前添加延迟 - 增加此值以减少API调用
            next_capture_delay = 10000  # 捕获间隔10秒
            self.app.after(next_capture_delay, self.trigger_next_capture)
        #总结：_analyze_screenshots() 是分析线程的核心，
        #负责：上传图像 → 调用模型 → 解析结果 → 更新日志/UI → 安排下一轮一气呵成、结构清晰、后台运行、自动轮询。

        #Q：为什么要把图像分析放在线程里，而不是主线程？？
        # 如果你把分析函数放在主线程，用户点击“暂停”按钮或者关闭窗口，界面会假死，
        # 因为程序正忙着上传图像/等AI模型返回，根本没时间响应事件循环。


    def _get_image_analysis(self, image_urls):
        #这里的自由度非常非常大！我们到时候要非常细致的讨论这个问题！
        """发送图像到Qwen-VL API并获取分析文本"""
        try:
            print("调用Qwen-VL API分析图像...")
            # 构建 messages 结构：详见文档！
            # 补充：
            # 在与大语言模型（LLM）交互时，构建 messages 结构是实现多轮对话的关键。messages 通常是一个数组，
            # 包含对话的上下文信息，确保模型能够理解当前对话的语境。以下是构建 messages 结构的通用格式和示例：

            # role: 指定消息的角色，常见值为：
            # system: 系统提示，用于设定模型的背景或行为。
            # user: 用户输入的内容。
            # assistant: 模型生成的回复。
            # content: 消息的具体内容。

            # 示例 ：复杂任务（带工具调用）
            # [
            #   {"role": "system", "content": "你是一个帮助用户完成任务的智能助手。"},
            #   {"role": "user", "content": "帮我生成一份会议纪要。"},
            #   {"role": "assistant", "content": "好的，请提供会议的主要内容和参与者信息。"},
            #   {"role": "user", "content": "会议讨论了项目进度，参与者有张三和李四。"}
            # ]
            # 注意事项
            # 上下文完整性: 确保对话历史中包含必要的信息，避免上下文丢失。
            # 内容简洁: 避免过多无关信息，节省上下文窗口。
            # 动态更新: 根据用户输入实时更新 messages，以保持对话的连贯性。

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
            
            #🚀 调用大模型 API 这个接口会把你的图像 + 文本送到大模型服务器，然后返回分析结果。
            completion = qwen_client.chat.completions.create(
                model="qwen-vl-max",
                messages=messages,
            )

            analysis_text = completion.choices[0].message.content
            #模型返回的结构是一个多层嵌套对象，
            #choices[0].message.content 是你要的分析结果，是一个中文字符串
            print(f"图像分析完成，分析长度: {len(analysis_text)} 字符")
            
            return analysis_text
            
        except Exception as e:
            error_msg = f"Qwen-VL API错误: {e}"
            print(error_msg)
            self.app.update_status(error_msg)
            return None
            #总结：_get_image_analysis() 
            # 发送图像 + 提示任务 → 等待 Qwen-VL 回答 → 返回分析文字结果
            # 是整个行为识别系统中，最智能的核心模块！

            #问题：SDK（软件开发工具包）和API（应用程序编程接口）
            # qwen_client.chat.completions.create(...) 这个格式是固定的吗？
            # 这是 Qwen 官方 SDK 中规定的标准调用格式



            
    def toggle_pause(self):
        """作用：用于“暂停/恢复”图像分析流程：切换分析循环的暂停状态"""
        #toggle:切换
        self.paused = not self.paused 
        # 每次调用都反转 True <-> False
        status = "已暂停分析" if self.paused else "已恢复分析"
        self.app.update_status(status)
        print(status)
        # 更新 UI 上的提示文字（告诉用户当前状态）
        # 同时输出到控制台，方便开发调试
        

        # 如果取消暂停，触发下一次捕获
        if not self.paused and not self.processing:
            self.app.after(500, self.trigger_next_capture)
            # 如果我们刚刚“取消了暂停”（即 self.paused = False）
            # 并且没有正在分析（self.processing = False）
            # 那就500 毫秒后自动触发一次新的图像分析循环
    
    def get_current_screenshot(self):
        """作用：获取当前摄像头画面的“最后一帧”图像（PIL 格式）：获取最近的摄像头图像"""
        return self.last_webcam_image
        #你在 _process_webcam() 中保存了摄像头最新帧到 self.last_webcam_image
    
    def _capture_screenshots(self, num_shots=4, interval=0.1):
        """ 作用：从摄像头连续捕获 num_shots 张截图 + 一张当前截图
            从摄像头捕获多个截图用于分析
            返回完整集合（用于分析）和一张当前截图（用于显示）"""
            # 默认捕获 4 张图用于行为分析
            # 每张图之间间隔 0.1 秒，模拟“动态图像”感觉
            # 同时再抓一张“当前最新帧”用于 UI 展示
        screenshots = []
        for i in range(num_shots):
            ret, frame = self.cap.read()
            if not ret:
                continue# 忽略错误帧
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            #Image.fromarray() 方法将一个 NumPy 数组（如 frame_rgb）转换为 PIL 图像对象。
            screenshots.append(img)
            time.sleep(interval)
            # 将图像从 OpenCV 的 BGR 转成 PIL 的 RGB 格式
            # 加入 screenshots 列表
            # 等 interval 秒（0.1s），再拍下一张
        
        # 再捕获一张当前帧专门用于显示
        ret, current_frame = self.cap.read()
        current_screenshot = None
        if ret:
            current_frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            current_screenshot = Image.fromarray(current_frame_rgb)
        # 这是额外拍的一张“当前帧”
        # 会作为当前 UI 显示用，不用于分析（分析用的是 screenshot 序列）


        if self.debug:
            print(f"已捕获 {len(screenshots)} 张截图用于分析和 1 张当前截图")
            
        return screenshots, current_screenshot
    
    def _upload_screenshots(self, screenshots):
        #它把你从摄像头捕获的图像上传到阿里云 OSS，让后面的图像分析模型（Qwen-VL）可以远程访问这些图片。
        """将截图上传到OSS并返回URL
        目标：把这些图片上传到阿里云 OSS
        最终返回：一个 URL 列表，供图像分析 API 使用
        """
        try:
            #1.连接 OSS 服务器
            #注意：配置应该是自己提前在项目中配置好的（或者在 .env 或 config 文件里写的）
            auth = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
            bucket = oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET)
            
            if self.debug:
                print(f"正在上传 {len(screenshots)} 张截图到OSS")
                
            oss_urls = []
            #把图像写入内存缓冲区
            for i, img in enumerate(screenshots):
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG')
                buffer.seek(0)
                # io.BytesIO() 创建一个内存中的“文件”
                # .save(..., format='JPEG') 把图片压缩成 JPEG 格式写进去
                # .seek(0) 让指针回到开头（准备读出来上传）
                # 💡 为什么不直接上传 img？
                # 因为 OSS 需要的是“字节流”，而不是 PIL 对象。

                #用时间戳 + 编号生成一个唯一文件名，防止覆盖
                #例如：screenshots/1722856741_0.jpg
                object_key = f"screenshots/{int(time.time())}_{i}.jpg"
                
                #上传到 OSS 并检查状态
                result = bucket.put_object(object_key, buffer)
                if result.status == 200:
                    #拼接出图片的公网地址
                    # 公网地址（Public URL）就是：
                    # 在任意一台联网的电脑或手机上，通过浏览器就可以访问的“图片链接”。
                    url = f"https://{OSS_BUCKET}.{OSS_ENDPOINT}/{object_key}"
                    #上传！
                    oss_urls.append(url)
                
                    if self.debug:
                        print(f"已上传图片 {i+1}: {url}")
                #基础的前后端知识：put_object() 把文件上传到 OSS 中返回 result，
                #里面有 HTTP 状态码状态码 200 说明上传成功->，404就是错误！
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

# ---------------- 主应用类 ----------------
class BehaviorVisualizationApp(ctk.CTk):
    def __init__(self):
        #self指的是类实例对象本身（注意：不是类本身）
        super().__init__()
        
        # 初始化系统组件
        self.setup_ui()
        #注意：self.setup_ui()是在调用函数而不是定义函数！函数在后面会定义！
        self.webcam_handler = WebcamHandler(self)
        # 把自己传进去
        # 把主窗口（self）当作参数传给 WebcamHandler，
        # 这样 WebcamHandler 就能通过 self.app 控制界面、更新状态、定时调用函数等。
        #“我这个主程序（BehaviorVisualizationApp）把自己交给 
        # WebcamHandler，以后你就可以随时喊我帮你做 UI 的事。”


        # 设置按键绑定
        self.setup_key_bindings()
        
        # 短暂延迟后启动摄像头
        self.after(1000, self.start_webcam)
        
        # 启动时间戳检查
        self.check_timestamp()
        
        # 设置观察历史
        self.observation_history = []
        
        # 标题和当前行为
        self.current_behavior = "未知"
    
    def start_webcam(self):
        """UI初始化后启动摄像头捕获"""
        if not self.webcam_handler.start():
            self.update_status("启动摄像头失败。请检查您的摄像头。")
    
    def setup_ui(self):
        """初始化用户界面->整个 GUI 界面的“搭建工程”"""
        self.title("行为监测与可视化系统")
        self.geometry("900x600")  # 修改界面尺寸为900x600
        
        # 设置暗色主题
        self.configure(fg_color="#1a1a1a")  # 深色背景
        
        # 配置网格configure:配置，设定/（核心！理解界面响应式的关键）
        # 在Tkinter中，grid_columnconfigure()和grid_rowconfigure()方法
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)  # 标题
        self.grid_rowconfigure(1, weight=1)  # 主要内容
        self.grid_rowconfigure(2, weight=0)  # 状态栏

        # grid_columnconfigure(index, weight=...) / grid_rowconfigure(index, weight=...)：
        # 用于设置第 index 列 / 行的 “权重”（拉伸比例）。
        # weight=0：表示该列 / 行尺寸固定，不会随窗口拉伸而变化（适合标题、状态栏等不需要缩放的部分）。
        # weight=1：表示该列 / 行会随窗口拉伸而按比例放大（适合主要内容区域，比如图表、摄像头画面等需要占满空间的部分）。
        # 这里的配置逻辑：
        # 整个界面只有 1 列（index=0），权重为 1，意味着窗口宽度变化时，这一列会填满整个宽度。
        # 界面分为 3 行：第 0 行（标题）和第 2 行（状态栏）固定高度，第 1 行（主要内容）会随窗口高度变化而拉伸，确保主要内容占满大部分空间。


        # 标题框架 grid:网格
        self.title_frame = ctk.CTkFrame(self, fg_color="#1a1a1a")
        # CTkFrame：框架组件，相当于一个 “容器”，用于将相关的 UI 元素（如下方的标题标签）组合在一起
        self.title_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))   
        # grid(row=0, column=0, ...)：将标题框架放在第 0 行、第 0 列（对应前面的网格配置）。
        # sticky="ew"：表示框架在水平方向（东 - 西，即左右）拉伸，填满所在的网格单元格（确保标题区域左右充满窗口）。
        # padx=10：框架左右的内边距（距离窗口边缘 10 像素），pady=(10, 0)：框架上下的内边距（上边 10 像素，下边 0 像素），避免内容贴边。

        
        self.title_label = ctk.CTkLabel(
            self.title_frame,
            text="行为监测与可视化系统",
            font=("Arial", 20, "bold"),#字体
            text_color="white"
        )
        self.title_label.pack(pady=10)
        





        # 主内容框架 （中间核心区域）
        self.main_frame = ctk.CTkFrame(self, fg_color="#1a1a1a")
        self.main_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        
        # 初始化行为可视化器（图表）
        self.behavior_visualizer = BehaviorVisualizer(self.main_frame)
        
        # 状态栏
        self.status_frame = ctk.CTkFrame(self, fg_color="#1a1a1a")
        self.status_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        
        # 当前行为显示
        self.behavior_label = ctk.CTkLabel(
            self.status_frame,
            text="当前行为: 未知",
            font=("Arial", 14, "bold"),
            text_color="white"
        )
        self.behavior_label.pack(side="left", padx=10, pady=5)
        #pack(side="left")：在状态栏中靠左放置，padx=10 是左右间距，pady=5 是上下间距。
        
        # 状态标签
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="就绪",
            font=("Arial", 12),
            text_color="white"
        )
        self.status_label.pack(side="right", padx=10, pady=5)
        
        #？？？？？？？self.status_frame,我有点不明白！
        # 他是ctk.CTkFrame()创建对象，又作为参数传到很多ctk.CTkLabel()里面？

        #答：你的观察很关键！在 CustomTkinter（和 Tkinter）中，所有组件都必须指定一个 “父容器”，
        # 而 self.status_frame 就是一个 “容器”
        # self.status_frame 本身是主窗口（self）的子组件（放在主窗口的第 2 行），负责 “状态栏” 这个区域。
        # 而 behavior_label、status_label 这些标签，又都是 status_frame 的子组件，
        # 所以它们会被 “约束” 在状态栏区域内，不会跑到标题区或主内容区。---->前端中的盒子结构！，下面同理！




        # 控制按钮
        self.controls_frame = ctk.CTkFrame(self.status_frame, fg_color="#1a1a1a")
        self.controls_frame.pack(side="top", fill="x")
        
        self.toggle_button = ctk.CTkButton(
            self.controls_frame,
            text="暂停分析",
            command=self.toggle_analysis,
            fg_color="#333333",
            text_color="white",
            hover_color="#555555"
        )
        self.toggle_button.pack(side="left", padx=10, pady=5)
        
        self.toggle_camera_button = ctk.CTkButton(
            self.controls_frame,
            text="显示/隐藏摄像头",
            command=self.toggle_camera,
            fg_color="#333333",
            text_color="white",
            hover_color="#555555"
        )
        self.toggle_camera_button.pack(side="left", padx=10, pady=5)
    





    def setup_key_bindings(self):
        """设置键盘快捷键"""
        self.bind("<space>", lambda e: self.toggle_analysis())#让摄像头分析逻辑暂停或继续。
        self.bind("<c>", lambda e: self.toggle_camera())#打开或关闭摄像头窗口。
        #这里的 self.bind() 是 tkinter 提供的方法，“当用户按某个键时，执行某个函数”
        #用于将键盘按键绑定到对应函数。绑定的函数用 lambda 包装是为了忽略事件参数 e
    
    def toggle_analysis(self):
        """切换分析循环的暂停状态"""
        self.webcam_handler.toggle_pause()
        #让摄像头行为识别 “暂停” 或 “继续”。取反！！！！！状态跌倒
        
        # 更新按钮文本
        new_text = "恢复分析" if self.webcam_handler.paused else "暂停分析"
        self.toggle_button.configure(text=new_text)
        #当你按下空格或点击按钮时，摄像头的分析状态会暂停或恢复，按钮的文字也会自动更新。


    
    def toggle_camera(self):
        """显示或隐藏摄像头窗口"""
        if self.webcam_handler.camera_window and not self.webcam_handler.camera_window.is_closed:
            self.webcam_handler.camera_window.on_closing()
        else:
            self.webcam_handler.create_camera_window()
    



    def add_behavior_data(self, timestamp, behavior_num, behavior_desc, analysis_text):
        """将检测到的行为添加到可视化和历史中"""
        # 添加到观察历史
        observation = {
            "timestamp": timestamp,
            "behavior_num": behavior_num,
            "behavior_desc": behavior_desc,
            "analysis": analysis_text
        }
        self.observation_history.append(observation)
        # 这是一个行为记录更新函数，当摄像头识别出行为后，它会告诉主程序：
        # timestamp：行为发生的时间（字符串或时间对象）
        # behavior_num：行为编号（比如 "1" 表示工作，"5" 表示玩手机）
        # behavior_desc：行为名称（如 "工作"、"玩手机"）
        # analysis_text：分析详情文本（可能是模型输出解释）


        
        # 限制历史长度，防止 observation_history 越来越长，导致内存爆炸；
        if len(self.observation_history) > 100:
            self.observation_history = self.observation_history[-100:]
            #保留最近100条记录
        
        # 添加到行为可视化器
        self.behavior_visualizer.add_behavior_data(timestamp, behavior_num, behavior_desc)
        
        # 更新当前行为显示，更新图表数据
        self.current_behavior = behavior_desc
        self.behavior_label.configure(text=f"当前行为: {behavior_desc}")
        
        # 根据行为更新UI颜色
        behavior_colors = {
            "1": "#4CAF50",  # 绿色表示工作
            "2": "#FFC107",  # 琥珀色表示吃东西
            "3": "#2196F3",  # 蓝色表示喝水
            "4": "#9C27B0",  # 紫色表示喝饮料
            "5": "#F44336",  # 红色表示玩手机
            "6": "#607D8B",  # 蓝灰色表示睡觉
            "7": "#795548"   # 棕色表示其他
        }
        
        try:
            # 根据行为设置标签文本颜色
            color = behavior_colors.get(behavior_num, "#000000")
            #get() 方法：如果找不到编号，就用默认颜色 #000000（黑色）
            self.behavior_label.configure(text_color=color)
        except Exception as e:
            print(f"更新UI颜色时出错: {e}")
    


    def update_status(self, text):
        """更新状态消息"""
        self.status_label.configure(text=text)
    
    def check_timestamp(self):
        """检查周期性更新（用于刷新图表）"""
        # 定期刷新可视化
        self.behavior_visualizer.refresh_charts()
        
        # 安排下一次检查
        self.after(30000, self.check_timestamp)  # 每30秒

# ---------------- 主函数 ----------------
def main():
    # 设置外观模式和默认主题
    ctk.set_appearance_mode("Dark")  # 设置为深色模式
    ctk.set_default_color_theme("blue")  # "blue", "green", "dark-blue"
    
    app = BehaviorVisualizationApp()
    app.protocol("WM_DELETE_WINDOW", lambda: quit_app(app))
    app.mainloop()

def quit_app(app):
    """干净地关闭应用程序"""
    # 停止所有线程
    if hasattr(app, 'webcam_handler'):
        app.webcam_handler.stop()
    
    if hasattr(app, 'behavior_visualizer'):
        app.behavior_visualizer.stop()
    
    # 关闭应用
    app.destroy()

if __name__ == "__main__":
    main()