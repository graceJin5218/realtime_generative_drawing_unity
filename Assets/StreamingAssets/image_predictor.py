import sys, os, io, socket, torch
from pydantic import BaseModel, Field
from PIL import Image
from typing import Literal, Optional
import numpy as np
import random
import torchvision.transforms as T

sys.path.append(os.path.join(os.path.dirname(__file__), "streamdiffusion"))
from utils.wrapper import StreamDiffusionWrapper

# 현재 스크립트가 위치한 디렉토리의 절대 경로를 가져옴
base_dir = os.path.dirname(os.path.abspath(__file__))

# 여러 위치에서 모델 파일을 찾도록 시도
model_filename = ""  # 移除硬编码的默认模型名称
possible_paths = [
    os.path.join(base_dir, "models", "Model", model_filename),  # 标准路径
    os.path.join("D:", "streamdiffusion_unity", "Assets", "StreamingAssets", "models", "Model", model_filename),  # 绝对路径
]

# 检查文件存在于哪个路径
found_model_path = None
for path in possible_paths:
    if model_filename and os.path.exists(path):
        found_model_path = path
        print(f"Found model file at: {found_model_path}")
        break

if found_model_path is None:
    # 如果找不到，使用默认路径并发出警告
    found_model_path = ""  # 设为空值，等待Unity客户端传入
    print(f"未找到默认模型文件，等待Unity客户端指定模型路径")

# 查找VAE模型文件
vae_possible_paths = [
    os.path.join(base_dir, "models", "VAE"),  # VAE文件夹
    os.path.join("D:", "streamdiffusion_unity", "Assets", "StreamingAssets", "models", "VAE"),  # 绝对路径
]

# 检查VAE文件夹是否存在且包含config.json
found_vae_path = None
for path in vae_possible_paths:
    config_path = os.path.join(path, "config.json")
    if os.path.exists(config_path):
        found_vae_path = path
        print(f"Found VAE folder with config.json at: {found_vae_path}")
        break

if found_vae_path is None:
    # 如果找不到VAE，使用默认HuggingFace模型
    found_vae_path = "madebyollin/taesd"
    print(f"WARNING: Could not find local VAE folder with config.json. Using HuggingFace model: {found_vae_path}")

# 查找LoRA模型
lora_filename = ""  # 不默认使用任何LoRA模型
lora_possible_paths = []  # 不自动扫描路径

# 检查LoRA文件存在于哪个路径
found_lora_path = None
# 初始不查找LoRA文件
print(f"LoRA model will be set by Unity client command")

# 第二个LoRA模型路径
found_lora_path2 = None
print(f"Second LoRA path not yet set. Will be configured later if needed.")

# 使用找到的路径
base_model = found_model_path  # 可能为空字符串
taesd_model = found_vae_path  # 使用找到的VAE路径
lora_model = found_lora_path  # 使用找到的LoRA文件路径或None
lora_model2 = found_lora_path2  # 第二个LoRA模型路径
pipeline_object = None
default_strength = 2.0  # 默认strength参数值
default_lora_scale = 0.85  # 默认第一个LoRA强度
default_lora_scale2 = 0.5  # 默认第二个LoRA强度
bypass_mode = False  # 是否启用绕过模式，直接返回输入图像而不进行AI处理
is_linear_space = False  # 颜色空间标志，True表示线性空间，False表示Gamma空间
is_in_prediction = False  # 进行推理中的标志，避免重复处理

print(f"初始模型路径设置为:\n基础模型={base_model if base_model else '未设置'}\nVAE={taesd_model}\nLoRA1={lora_model}\nLoRA2={lora_model2}")
print(f"默认强度参数: {default_strength}")
print(f"默认LoRA强度: LoRA1={default_lora_scale}, LoRA2={default_lora_scale2}")

class Pipeline:
    def __init__(self, w: int, h: int, seed: int, device: torch.device, torch_dtype: torch.dtype,
                 use_vae: bool, use_lora: bool, gc_mode: Literal["img2img", "txt2img"], acc_mode: str,
                 positive_prompt: str, negative_prompt: str = "", preloaded_pipe=None, model_path=None, lora_dict=None,
                 cfg_type: str = "none", delta: float = 0.8, do_add_noise: bool = True, enable_similar_image_filter: bool = True,
                 similar_image_filter_threshold: float = 0.6, similar_image_filter_max_skip_frame: int = 10):
        # 使用传入的特定路径
        actual_model_path = model_path if model_path else base_model
        print(f"Initializing pipeline with model path: {actual_model_path}")
        
        # 设置类变量，供prepare方法使用
        self.delta = delta
        self.cfg_type = cfg_type
        self.guidance_scale = 1.0  # 默认值与main.py一致
        
        # 保存其他参数
        self.do_add_noise = do_add_noise
        self.enable_similar_image_filter = enable_similar_image_filter
        self.similar_image_filter_threshold = similar_image_filter_threshold
        self.similar_image_filter_max_skip_frame = similar_image_filter_max_skip_frame
        
        if not os.path.exists(actual_model_path):
            print(f"Error: Model path does not exist: {actual_model_path}")
            return
        
        # 确定VAE路径
        vae_path = None
        if use_vae:
            vae_path = "madebyollin/taesd"  # 默认使用HuggingFace上的模型
            if taesd_model and os.path.exists(taesd_model) and os.path.isdir(taesd_model):
                # 用本地目录中的VAE替换
                print(f"Using local VAE folder: {taesd_model}")
                vae_path = taesd_model
        
        # 处理lora_dict
        if use_lora and lora_dict is not None:
            print(f"LoRA will be used: {lora_dict}")
            
        # 创建StreamDiffusionWrapper
        if not preloaded_pipe:
            self.stream = StreamDiffusionWrapper(
                model_id_or_path=actual_model_path,
                t_index_list=[15, 25],
                lora_dict=lora_dict,
                vae_id=vae_path,
                mode=gc_mode,
                seed=seed,
                cfg_type=cfg_type,
                use_safety_checker=False,
                engine_dir="engines",
                # 以下参数需要从StreamDiffusionWrapper初始化中移除
                # do_add_noise和enable_similar_image_filter等，因为StreamDiffusionWrapper不接受这些参数
                frame_buffer_size=1,
                width=w, 
                height=h,
                warmup=10,
                acceleration=acc_mode,
                use_denoising_batch=True,
                device=device,
                dtype=torch_dtype,
                output_type="pil"  # 修改为tensor类型输出
            )
        else:
            # 使用预加载的pipeline
            self.stream = StreamDiffusionWrapper(
                model_id_or_path=actual_model_path,
                t_index_list=[15, 25],
                lora_dict=lora_dict,
                vae_id=vae_path,
                mode=gc_mode,
                seed=seed,
                cfg_type=cfg_type,
                use_safety_checker=False,
                engine_dir="engines",
                # 以下参数需要从StreamDiffusionWrapper初始化中移除
                # do_add_noise和enable_similar_image_filter等，因为StreamDiffusionWrapper不接受这些参数
                frame_buffer_size=1,
                width=w, 
                height=h,
                warmup=10,
                acceleration=acc_mode,
                use_denoising_batch=True,
                device=device,
                dtype=torch_dtype,
                output_type="pil"  # 修改为tensor类型输出
            )
        # 初始化阶段不调用prepare，将在loadPipeline函数中调用

    def prepare(self, prompt, negative_prompt, target_guidance_scale=None):
        try:
            print(f"准备生成，提示词: '{prompt}', 负面提示词: '{negative_prompt}'")
            delta_value = self.delta if hasattr(self, 'delta') else 0.8
            strength_value = 2.0  # 默认值
            
            # 获取guidance_scale值，优先使用传入的参数
            guidance_scale_value = target_guidance_scale if target_guidance_scale is not None else self.guidance_scale
            
            # 获取全局strength值
            global default_strength
            if default_strength is not None:
                strength_value = default_strength
            
            print(f"使用参数: 引导规模={guidance_scale_value}, delta={delta_value}, strength={strength_value}")
            
            # 准备参数字典，包含StreamDiffusionWrapper.prepare支持的参数
            prepare_params = {
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                'guidance_scale': guidance_scale_value,  # 默认值1.0，与main.py一致
                'delta': delta_value,
                'num_inference_steps': 50
            }
            
            # 调用prepare方法
            self.stream.prepare(**prepare_params)
            
            # 设置其他属性（如果StreamDiffusionWrapper支持）
            if hasattr(self.stream, 'stream'):
                if hasattr(self.stream.stream, 'do_add_noise') and hasattr(self, 'do_add_noise'):
                    self.stream.stream.do_add_noise = self.do_add_noise
                    print(f"设置do_add_noise: {self.do_add_noise}")
                
                if hasattr(self.stream.stream, 'enable_similar_image_filter') and hasattr(self, 'enable_similar_image_filter'):
                    self.stream.stream.enable_similar_image_filter = self.enable_similar_image_filter
                    print(f"设置enable_similar_image_filter: {self.enable_similar_image_filter}")
                
                if hasattr(self.stream.stream, 'similar_image_filter_threshold') and hasattr(self, 'similar_image_filter_threshold'):
                    self.stream.stream.similar_image_filter_threshold = self.similar_image_filter_threshold
                    print(f"设置similar_image_filter_threshold: {self.similar_image_filter_threshold}")
                
                if hasattr(self.stream.stream, 'similar_image_filter_max_skip_frame') and hasattr(self, 'similar_image_filter_max_skip_frame'):
                    self.stream.stream.similar_image_filter_max_skip_frame = self.similar_image_filter_max_skip_frame
                    print(f"设置similar_image_filter_max_skip_frame: {self.similar_image_filter_max_skip_frame}")
            
            print(f"生成器准备完成")
        except Exception as e:
            print(f"准备生成器时出错: {e}")
            import traceback
            traceback.print_exc()

    def predict(self, input_image: Image.Image, new_prompt: Optional[str] = None) -> Image.Image:
        try:
            print(f"Input image size: {input_image.size}, mode: {input_image.mode}")
            
            # 1. 먼저 모든 이미지를 RGB로 변환
            if input_image.mode != "RGB":
                print(f"Converting image from {input_image.mode} to RGB")
                if input_image.mode == "RGBA":
                    # RGBA의 경우 흰색 배경으로 합성
                    background = Image.new("RGB", input_image.size, (255, 255, 255))
                    background.paste(input_image, mask=input_image.split()[3] if len(input_image.split()) > 3 else None)
                    input_image = background
                elif input_image.mode == "L":
                    # 그레이스케일을 RGB로
                    input_image = input_image.convert("RGB")
                elif input_image.mode == "P":
                    # 팔레트 모드를 RGB로
                    input_image = input_image.convert("RGB")
                else:
                    # 기타 모드도 RGB로 변환
                    input_image = input_image.convert("RGB")
                print(f"Converted to RGB mode")

            # Transform 정의 (3채널용)
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

            # 디버깅 이미지 저장
            try:
                input_copy = input_image.copy()
                input_debug_path = "debug_input.png"
                input_copy.save(input_debug_path)
                print(f"보존 입력 이미지 대상: {input_debug_path}")
            except Exception as e:
                print(f"저장 디버그 이미지 실패: {e}")

            # numpy 배열로 변환 (RGB 확인)
            arr_input = np.array(input_image)
            print(f"Array shape: {arr_input.shape}")  # (height, width, 3) 이어야 함
            
            # 채널 수 확인
            if len(arr_input.shape) == 3 and arr_input.shape[2] == 4:
                print("WARNING: Still has 4 channels, removing alpha")
                arr_input = arr_input[:, :, :3]
            elif len(arr_input.shape) == 2:
                print("WARNING: Grayscale detected, converting to RGB")
                arr_input = np.stack([arr_input] * 3, axis=2)
                
            print(f"입력 이미지 평균 RGB 값: R={arr_input[:,:,0].mean():.2f}, G={arr_input[:,:,1].mean():.2f}, B={arr_input[:,:,2].mean():.2f}")

            global bypass_mode, is_linear_space

            # 공백 이미지 처리
            is_blank = arr_input.mean() < 10
            if is_blank:
                print("검색된 공백 입력 이미지, 테스트 이미지 생성 대체")
                test_array = np.zeros((512, 512, 3), dtype=np.uint8)
                for y in range(512):
                    for x in range(512):
                        if x < 170:
                            test_array[y, x] = [200, 100, 50]
                        elif x < 340:
                            test_array[y, x] = [50, 200, 100]
                        else:
                            test_array[y, x] = [100, 50, 200]
                input_image = Image.fromarray(test_array, mode='RGB')  # 명시적으로 RGB 모드 지정
                arr_input = test_array

            # 색공간 처리
            if is_linear_space:
                print("검색된 선형 색공간 입력, Gamma 공간으로 변환")
                
                # 3채널인지 확인
                if arr_input.shape[2] != 3:
                    print(f"ERROR: Expected 3 channels, got {arr_input.shape[2]}")
                    arr_input = arr_input[:, :, :3] if arr_input.shape[2] > 3 else np.stack([arr_input[:, :, 0]] * 3, axis=2)
                
                arr_float = arr_input.astype(np.float32) / 255.0
                arr_gamma = np.power(arr_float, 1/2.2) * 255.0
                arr_gamma = np.clip(arr_gamma, 0, 255).astype(np.uint8)
                
                print(f"Gamma array shape: {arr_gamma.shape}")  # 디버그용
                print(f"변환 후 평균 RGB 값: R={arr_gamma[:,:,0].mean():.2f}, G={arr_gamma[:,:,1].mean():.2f}, B={arr_gamma[:,:,2].mean():.2f}")
                
                # PIL Image 생성 시 명시적으로 RGB 모드 지정
                gamma_image = Image.fromarray(arr_gamma, mode='RGB')
                
                # 한 번 더 확인
                if gamma_image.mode != "RGB":
                    print(f"WARNING: gamma_image is {gamma_image.mode}, converting to RGB")
                    gamma_image = gamma_image.convert("RGB")
                
                print(f"Gamma image mode: {gamma_image.mode}, size: {gamma_image.size}")
                    
                try:
                    gamma_image.save("debug_gamma_converted.png")
                    print("Gamma 변환 이미지를 debug_gamma_converted.png에 저장")
                except Exception as e:
                    print(f"Gamma 이미지 저장 실패: {e}")
                
                # Transform 적용
                image_tensor = transform(gamma_image).unsqueeze(0)
            else:
                print("비선형 공간, 입력 이미지 직접 처리")
                # 한 번 더 RGB 확인
                if input_image.mode != "RGB":
                    print(f"Final check: converting {input_image.mode} to RGB")
                    input_image = input_image.convert("RGB")
                
                image_tensor = transform(input_image).unsqueeze(0)

            print(f"Tensor shape after transform: {image_tensor.shape}")  # [1, 3, H, W] 이어야 함

            # 장치 및 dtype 일관되게 적용
            if torch.cuda.is_available():
                image_tensor = image_tensor.to(device="cuda", dtype=torch.float16)
            else:
                image_tensor = image_tensor.to(self.stream.device if hasattr(self, 'stream') else "cpu")

            # 프롬프트 업데이트
            if new_prompt:
                print(f"프롬프트 사용: {new_prompt}")
            try:
                self.stream.stream.update_prompt(new_prompt if new_prompt else "")
                print("프롬프트 업데이트 성공")
            except Exception as e:
                print(f"프롬프트 업데이트 실패: {e}")

            # 예열 및 추론
            try:
                # 예열 단계
                for _ in range(getattr(self.stream, "batch_size", 1) - 1):
                    print(f"예열 단계 실행 {_+1}/{getattr(self.stream, 'batch_size', 1)-1}")
                    self.stream(image=image_tensor, prompt=new_prompt)
                
                print("최종 추론 단계 실행")
                output_tensor = self.stream(image=image_tensor, prompt=new_prompt)

                # 출력 텐서 처리 및 PIL 이미지로 변환
                if isinstance(output_tensor, torch.Tensor):
                    print(f"텐서 출력을 PIL 이미지로 변환, 형태: {output_tensor.shape}")
                    
                    # 배치 차원 제거
                    if output_tensor.dim() == 4:  # [batch, channel, height, width]
                        output_tensor = output_tensor.squeeze(0)  # 배치 차원 제거
                    
                    # 값 범위를 [0,1]로 클램핑하고 CPU로 이동
                    output_tensor = output_tensor.clamp(0, 1).cpu().float()
                    
                    # PIL 이미지로 변환
                    if output_tensor.shape[0] == 3:  # [channel, height, width]
                        output_array = output_tensor.permute(1, 2, 0).numpy()  # [height, width, channel]
                        output_array = (output_array * 255).astype(np.uint8)
                        output_image = Image.fromarray(output_array, mode='RGB')  # 명시적으로 RGB 지정
                        print(f"PIL 이미지로 변환 성공, 크기: {output_image.size}, 모드: {output_image.mode}")
                        
                        # RGB 모드 확인
                        if output_image.mode != "RGB":
                            print(f"이미지 모드를 {output_image.mode}에서 RGB로 변환")
                            output_image = output_image.convert("RGB")
                        
                        # 출력 이미지 RGB 값 분석
                        arr_output = np.array(output_image)
                        print(f"출력 이미지 평균 RGB 값: R={arr_output[:,:,0].mean():.2f}, G={arr_output[:,:,1].mean():.2f}, B={arr_output[:,:,2].mean():.2f}")
                        
                        # 디버그 출력 저장
                        try:
                            output_image.save("debug_final_output.png")
                            print("최종 출력을 debug_final_output.png에 저장")
                        except Exception as e:
                            print(f"디버그 최종 출력 저장 실패: {e}")
                        
                        return output_image
                        
                    else:
                        print(f"예상치 못한 텐서 형태: {output_tensor.shape}")
                        # 폴백: 테스트 이미지 생성
                        fallback_image = Image.new("RGB", (512, 512), (0, 0, 255))  # 파란색 테스트 이미지
                        try:
                            from PIL import ImageDraw
                            draw = ImageDraw.Draw(fallback_image)
                            draw.text((100, 100), "형태 오류", fill=(255, 255, 255))
                        except Exception:
                            pass
                        return fallback_image
                        
                else:
                    # 이미 PIL 이미지인 경우
                    print(f"출력이 이미 PIL 이미지, 크기: {output_tensor.size}, 모드: {output_tensor.mode}")
                    if output_tensor.mode != "RGB":
                        output_tensor = output_tensor.convert("RGB")
                    return output_tensor

            except Exception as e:
                print(f"StreamDiffusion 추론 과정 중 오류: {e}")
                import traceback
                traceback.print_exc()
                raise

        except Exception as e:
            print(f"predict에서 오류: {e}")
            import traceback
            traceback.print_exc()
            
            # 폴백 테스트 이미지 반환
            print("폴백 테스트 이미지 생성")
            fallback_image = Image.new("RGB", (512, 512), (255, 0, 0))  # 빨간색 테스트 이미지
            try:
                from PIL import ImageDraw
                draw = ImageDraw.Draw(fallback_image)
                draw.rectangle((100, 100, 400, 400), fill=(0, 255, 0))  # 녹색 사각형
                draw.ellipse((200, 200, 300, 300), fill=(0, 0, 255))  # 파란색 원
                draw.text((50, 50), "오류 발생", fill=(255, 255, 255))
            except Exception:
                pass
            return fallback_image


# Re-add setModelPaths function
def setModelPaths(base_m: str, tiny_vae_m: str, lcm_lora_m: str, lcm_lora_m2: str = None):
    global base_model
    global taesd_model
    global lora_model
    global lora_model2
    
    print(f"接收到的路径: 基础模型={base_m}, VAE={tiny_vae_m}, LoRA1={lcm_lora_m}, LoRA2={lcm_lora_m2}")
    
    # 更新基础模型路径
    if base_m and os.path.exists(base_m):
        base_model = base_m
        print(f"基础模型路径已更新为: {base_model}")
    else:
        print(f"警告: 未提供有效的基础模型路径或文件不存在: {base_m}")
        if not base_model:  # 如果之前的base_model为空
            print("错误: 没有有效的基础模型路径，无法加载模型")
    
    # 更新VAE模型路径
    if tiny_vae_m:
        if os.path.exists(tiny_vae_m) and os.path.exists(os.path.join(tiny_vae_m, "config.json")):
            taesd_model = tiny_vae_m
            print(f"VAE模型路径已更新为本地文件夹: {taesd_model}")
        elif tiny_vae_m.startswith("madebyollin"):
            taesd_model = tiny_vae_m
            print(f"VAE模型设置为HuggingFace模型: {taesd_model}")
        else:
            print(f"VAE路径无效或缺少config.json: {tiny_vae_m}")
            config_exists = False
            # 检查是否有效的本地VAE文件夹路径
            for possible_path in [tiny_vae_m, os.path.dirname(tiny_vae_m)]:
                config_path = os.path.join(possible_path, "config.json")
                if os.path.exists(config_path):
                    taesd_model = possible_path
                    print(f"在目录中找到config.json: {taesd_model}")
                    config_exists = True
                    break
            
            if not config_exists:
                print("切换到HuggingFace模型")
                taesd_model = "madebyollin/taesd"
                print(f"现在使用HuggingFace VAE模型: {taesd_model}")
    
    # 检查LoRA模型路径
    if lcm_lora_m:
        if os.path.exists(lcm_lora_m):
            lora_model = lcm_lora_m
            print(f"LoRA1模型路径已更新为: {lora_model}")
        else:
            print(f"LoRA1模型路径未找到: {lcm_lora_m}")
            lora_model = None  # 路径无效时设为None
    else:
        print("LoRA1模型路径为空，设置为None")
        lora_model = None  # 空路径时设为None
        
    # 检查第二个LoRA模型路径
    if lcm_lora_m2:
        if os.path.exists(lcm_lora_m2):
            lora_model2 = lcm_lora_m2
            print(f"LoRA2模型路径已更新为: {lora_model2}")
        else:
            print(f"LoRA2模型路径未找到: {lcm_lora_m2}")
            lora_model2 = None  # 路径无效时设为None
    else:
        print("LoRA2模型路径为空，设置为None")
        lora_model2 = None  # 空路径时设为None
    
    print(f"最终路径: 基础模型={base_model}, VAE={taesd_model}, LoRA1={lora_model}, LoRA2={lora_model2}")

def loadPipeline(w: int, h: int, seed: int, use_vae: bool, use_lora: bool,
                 acc_mode: str, positive_prompt: str, negative_prompt: str, strength: float = None, 
                 lora_scale: float = None, lora_scale2: float = None,
                 delta: float = 0.8, do_add_noise: bool = True,
                 enable_similar_image_filter: bool = True,
                 similar_image_filter_threshold: float = 0.6,
                 similar_image_filter_max_skip_frame: int = 10,
                 guidance_scale: float = 1.0):
    # 使用全局默认值或传入的值
    global default_strength, default_lora_scale, default_lora_scale2, pipeline_object
    
    # 检查基础模型路径是否有效
    if not base_model:
        print("错误: 没有有效的基础模型路径，无法加载Pipeline")
        return False
    
    if not os.path.exists(base_model):
        print(f"错误: 基础模型文件不存在: {base_model}")
        return False
        
    if strength is None:
        strength = default_strength
        print(f"使用默认强度值: {strength}")
    else:
        # 如果提供了新的strength值，更新全局默认值
        default_strength = strength
        print(f"更新默认强度为: {default_strength}")
    
    # 处理LoRA强度值，确保lora_scale不为None
    if lora_scale is None:
        lora_scale = default_lora_scale
        print(f"使用默认LoRA1强度: {lora_scale}")
    elif isinstance(lora_scale, (int, float)) and lora_scale <= 0:
        lora_scale = default_lora_scale
        print(f"无效的LoRA1强度 (<=0)，使用默认值: {lora_scale}")
    else:
        try:
            lora_scale = float(lora_scale)  # 尝试转换为float
            default_lora_scale = lora_scale
            print(f"更新默认LoRA1强度为: {default_lora_scale}")
        except (TypeError, ValueError):
            lora_scale = default_lora_scale
            print(f"无效的LoRA1强度格式，使用默认值: {lora_scale}")
        
    # 处理第二个LoRA强度值，确保lora_scale2不为None
    if lora_scale2 is None:
        lora_scale2 = default_lora_scale2
        print(f"使用默认LoRA2强度: {lora_scale2}")
    elif isinstance(lora_scale2, (int, float)) and lora_scale2 <= 0:
        lora_scale2 = default_lora_scale2
        print(f"无效的LoRA2强度 (<=0)，使用默认值: {lora_scale2}")
    else:
        try:
            lora_scale2 = float(lora_scale2)  # 尝试转换为float
            default_lora_scale2 = lora_scale2
            print(f"更新默认LoRA2强度为: {default_lora_scale2}")
        except (TypeError, ValueError):
            lora_scale2 = default_lora_scale2
            print(f"无效的LoRA2强度格式，使用默认值: {lora_scale2}")
    
    # 确保提示词为空时使用空字符串，而不是预设值
    if positive_prompt is None:
        positive_prompt = ""
    if negative_prompt is None:
        negative_prompt = ""
        
    print(f"加载Pipeline，模型路径: {base_model}")
    print(f"基础模型文件存在: {os.path.exists(base_model)}")
    print(f"VAE模型: {taesd_model}")
    print(f"LoRA1模型: {lora_model}, 存在: {os.path.exists(lora_model) if lora_model else False}")
    print(f"LoRA2模型: {lora_model2}, 存在: {os.path.exists(lora_model2) if lora_model2 else False}")
    print(f"使用VAE: {use_vae}, 使用LoRA: {use_lora}")
    print(f"LoRA1强度: {lora_scale}, LoRA2强度: {lora_scale2}")
    print(f"强度: {strength}")
    print(f"提示词: '{positive_prompt}'")
    print(f"负面提示词: '{negative_prompt}'")
    print(f"Delta: {delta}, 添加噪声: {do_add_noise}")
    print(f"图像相似度过滤: {enable_similar_image_filter}, 阈值: {similar_image_filter_threshold}, 最大跳帧: {similar_image_filter_max_skip_frame}")
    print(f"引导尺度: {guidance_scale}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16
    
    try:
        print("Initializing pipeline with standard method")
        
        # 决定是否使用LoRA
        use_lcm_lora_value = False
        if use_lora and lora_model is not None and os.path.exists(lora_model):
            print(f"LoRA will be used: {lora_model}")
            use_lcm_lora_value = True
            
        # 创建LoRA字典，使用两个LoRA模型
        lora_dict_to_use = {}
        
        # 添加第一个LoRA
        if use_lora and lora_model is not None and os.path.exists(lora_model):
            lora_dict_to_use[lora_model] = lora_scale
            print(f"Using LoRA1 with scale: {lora_scale}")
            
        # 添加第二个LoRA（如果存在）
        if use_lora and lora_model2 is not None and os.path.exists(lora_model2):
            lora_dict_to_use[lora_model2] = lora_scale2
            print(f"Using LoRA2 with scale: {lora_scale2}")
            
        if not lora_dict_to_use:
            lora_dict_to_use = None
            print("No valid LoRA models found")
        else:
            print(f"Using LoRA with dictionary: {lora_dict_to_use}")
        
        # 使用传入的参数初始化Pipeline
        pipeline_object = Pipeline(
            w=w, h=h, seed=seed, device=device, torch_dtype=torch_dtype, 
            use_vae=True,  # 始终使用VAE，无论传入参数如何
            use_lora=use_lora,  # 根据传入参数决定是否使用LoRA
            gc_mode="img2img",
            acc_mode=acc_mode, 
            positive_prompt=positive_prompt, 
            negative_prompt=negative_prompt, 
            model_path=base_model,
            lora_dict=lora_dict_to_use,  # 传递LoRA字典
            cfg_type="none",  # 使用none类型的CFG
            delta=delta,  # 设置delta参数
            do_add_noise=do_add_noise,  # 是否添加噪声
            enable_similar_image_filter=enable_similar_image_filter,  # 启用相似图像过滤
            similar_image_filter_threshold=similar_image_filter_threshold,  # 设置相似度阈值
            similar_image_filter_max_skip_frame=similar_image_filter_max_skip_frame  # 设置最大跳过帧数
        )

        # 在Pipeline初始化后设置strength参数
        try:
            print(f"调用prepare方法，提示词: '{positive_prompt}'")
            pipeline_object.prepare(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                target_guidance_scale=guidance_scale  # 使用传入的guidance_scale
            )
            print(f"Stream已准备好，可以进行生成")
            return True
        except Exception as e:
            print(f"调用prepare方法时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    except Exception as e:
        print(f"ERROR IN LOADING PIPELINE: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    # 最终检查pipeline_object是否成功创建
    if pipeline_object is None:
        print("Pipeline创建失败，结果为None")
        return False
    
    print("Pipeline加载成功")
    return True

def runPipeline(input_bytes, new_prompt: str):
    output_io = io.BytesIO()
    if pipeline_object is None:
        # pipeline이 비어있으면 테스트 이미지 반환
        print("pipeline이 비어있음, 테스트 이미지 반환")
        test_image = Image.new("RGB", (512, 512), (255, 100, 100))  # 빨간색 테스트 이미지
        try:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(test_image)
            draw.rectangle((100, 100, 400, 400), fill=(100, 255, 100))  # 녹색 사각형
            draw.ellipse((200, 200, 300, 300), fill=(100, 100, 255))  # 파란색 원
        except Exception:
            pass
        test_image.save(output_io, format="PNG")
        return output_io.getvalue()
    else:
        try:
            # 프롬프트가 None이 아닌지 확인
            if new_prompt is None:
                new_prompt = ""
                
            try:
                input_image = Image.open(io.BytesIO(input_bytes))
                print(f"입력 이미지 열기 성공, 크기: {input_image.size}, 모드: {input_image.mode}")
                
                # 바이패스 모드 확인
                global bypass_mode
                if bypass_mode:
                    print("바이패스 모드 활성화, 원본 입력 이미지 직접 반환")
                    
                    # 원본 이미지의 RGB 값 분석
                    arr = np.array(input_image)
                    print(f"원본 이미지 평균 RGB 값: R={arr[:,:,0].mean():.2f}, G={arr[:,:,1].mean():.2f}, B={arr[:,:,2].mean():.2f}")
                    
                    # 중요: 원본 입력 이미지를 직접 반환, 변환이나 처리 없이
                    input_bytes_io = io.BytesIO()
                    input_image.save(input_bytes_io, format="PNG")
                    print(f"바이패스 모드: 원본 이미지 직접 반환, 데이터 크기: {input_bytes_io.tell()} 바이트")
                    return input_bytes_io.getvalue()
                
                # 디버그용 입력 이미지 저장
                try:
                    debug_input = input_image.copy()
                    debug_input.save("debug_input_from_unity.png")
                    print("Unity에서 전송된 입력 이미지 저장")
                except Exception as e:
                    print(f"디버그 입력 저장 실패: {e}")
                    
            except Exception as e:
                print(f"입력 이미지 열기 실패: {e}")
                # 간단한 테스트 이미지로 대체
                input_image = Image.new("RGB", (512, 512), (0, 255, 0))  # 녹색 테스트 이미지
                print("녹색 테스트 이미지를 대체로 생성")
            
            # 정상적인 이미지 생성 흐름
            print("pipeline_object.predict 호출하여 이미지 처리")
            
            # 중복 생성 방지를 위한 전역 플래그 설정
            global is_in_prediction
            is_in_prediction = True
            
            try:
                # predict 메서드는 이제 PIL 이미지를 반환
                output_image = pipeline_object.predict(input_image, new_prompt)
                is_in_prediction = False
                print("예측 완료")
                
                # output_image는 이미 PIL Image 객체
                if isinstance(output_image, Image.Image):
                    print(f"출력이 PIL 이미지, 크기: {output_image.size}, 모드: {output_image.mode}")
                    
                    # RGB 모드 확인
                    if output_image.mode != "RGB":
                        print(f"이미지 모드를 {output_image.mode}에서 RGB로 변환")
                        output_image = output_image.convert("RGB")
                    
                    # 출력 이미지의 RGB 값 분석
                    arr_output = np.array(output_image)
                    print(f"출력 이미지 평균 RGB 값: R={arr_output[:,:,0].mean():.2f}, G={arr_output[:,:,1].mean():.2f}, B={arr_output[:,:,2].mean():.2f}")
                    
                    # 출력 이미지가 너무 어두운지 확인
                    is_dark = arr_output.mean() < 30  # 평균 밝기가 30/255 미만
                    if is_dark:
                        print("경고: 출력 이미지가 너무 어두움, 대체 이미지 생성")
                        # 대체 컬러 출력 생성
                        output_image = Image.new("RGB", (512, 512), (128, 128, 128))  # 회색 배경
                        try:
                            from PIL import ImageDraw
                            draw = ImageDraw.Draw(output_image)
                            
                            # 컬러풀한 줄무늬 그리기
                            for y in range(0, 512, 16):
                                color = ((y*3) % 256, (y*5) % 256, (y*7) % 256)
                                draw.rectangle((0, y, 512, y+8), fill=color)
                                
                            # 텍스트 추가
                            draw.text((100, 240), "생성 실패 - 어두운 이미지 대체", fill=(255, 255, 255))
                        except Exception as e:
                            print(f"대체 이미지 그리기 실패: {e}")
                    
                    # PNG 형식으로 저장하여 색상 손실 방지
                    output_image.save(output_io, format="PNG")
                    print(f"이미지 저장 후 데이터 크기: {output_io.tell()} 바이트")
                    
                    # 최종 출력 디버그 저장
                    try:
                        output_image.save("debug_final_output.png")
                        print("최종 출력을 debug_final_output.png에 저장")
                    except Exception as e:
                        print(f"디버그 최종 출력 저장 실패: {e}")
                    
                    # 출력 데이터 유효성 확인
                    if output_io.tell() == 0:
                        print("경고: 출력 데이터 크기가 0, 자주색 테스트 이미지 생성")
                        test_image = Image.new("RGB", (512, 512), (255, 0, 255))  # 자주색 테스트 이미지
                        try:
                            from PIL import ImageDraw
                            draw = ImageDraw.Draw(test_image)
                            draw.text((100, 240), "빈 데이터 대체", fill=(255, 255, 255))
                        except Exception:
                            pass
                        output_io = io.BytesIO()
                        test_image.save(output_io, format="PNG")
                    
                    return output_io.getvalue()
                
                else:
                    print(f"예상치 못한 출력 타입: {type(output_image)}")
                    # 폴백: 파란색 테스트 이미지
                    fallback_image = Image.new("RGB", (512, 512), (0, 0, 255))
                    try:
                        from PIL import ImageDraw
                        draw = ImageDraw.Draw(fallback_image)
                        draw.text((100, 240), "타입 오류", fill=(255, 255, 255))
                    except Exception:
                        pass
                    fallback_image.save(output_io, format="PNG")
                    return output_io.getvalue()
                    
            except Exception as e:
                is_in_prediction = False
                print(f"처리 중 오류: {e}")
                import traceback
                traceback.print_exc()
                raise  # 외부 처리를 위해 예외 다시 발생
                
        except Exception as e:
            print(f"runPipeline에서 오류: {e}")
            import traceback
            traceback.print_exc()
            
            # 오류 표시용 주황색 테스트 이미지 생성
            test_image = Image.new("RGB", (512, 512), (255, 165, 0))  # 주황색 테스트 이미지
            try:
                from PIL import ImageDraw
                draw = ImageDraw.Draw(test_image)
                draw.text((100, 240), f"오류: {str(e)[:50]}", fill=(0, 0, 0))
            except Exception:
                pass
            test_io = io.BytesIO()
            test_image.save(test_io, format="PNG")
            print("오류 대체로 주황색 테스트 이미지 반환")
            return test_io.getvalue()

def processData(client_socket, data):
    global pipeline_object, bypass_mode, is_linear_space
    try:
        print(f"Processing data, length: {len(data)}")
        
        # 安全地解码数据，处理可能的编码错误
        try:
            data_str = data.decode('utf-8')
            print(f"Data decoded successfully as UTF-8: {data_str[:50]}..." if len(data_str) > 50 else data_str)
        except UnicodeDecodeError:
            # 如果出现解码错误，跳过解码步骤，直接将数据视为二进制
            print(f"Failed to decode data as UTF-8, treating as binary")
            
        data = data.strip(b"|start|").strip(b"|end|")
        parts = data.split(b"||")
        print(f"Split data into {len(parts)} parts")

        use_vae = True
        use_lora = True
        width = 512
        height = 512
        seed = -1  # 设置为-1，这样如果没有正确接收种子参数就会很明显
        strength = None  # 初始化为None，会使用全局default_strength
        lora_scale = None  # 第一个LoRA强度，默认使用全局default_lora_scale
        lora_scale2 = None  # 第二个LoRA强度，默认使用全局default_lora_scale2
        # 新增参数，设置默认值与main.py一致
        delta = 0.8  # 默认delta值
        do_add_noise = True  # 默认添加噪声
        enable_similar_image_filter = True  # 默认启用相似图像过滤
        similar_image_filter_threshold = 0.6  # 默认相似度阈值
        similar_image_filter_max_skip_frame = 10  # 默认最大跳过帧数
        guidance_scale = 1.0  # 默认guidance_scale，与main.py一致
        command_state = 0
        command = ""
        acc_mode = "tensorrt"
        prompt = ""
        neg_prompt = ""
        base_m = base_model
        lora_m = None  # 第一个LoRA模型路径
        lora_m2 = None  # 第二个LoRA模型路径
        image = io.BytesIO()

        for i in range(0, len(parts), 2):
            if i + 1 >= len(parts):
                break
                
            # 安全地解码键
            try:
                key = parts[i].decode('utf-8')
            except UnicodeDecodeError:
                print(f"Failed to decode key at index {i}, skipping this pair")
                continue
                
            value = parts[i + 1]
            
            # 打印接收到的关键参数
            if key in ["command", "base_model", "taesd_model", "lora_model", "lora_model2", "prompt", "acceleration", "strength", "lora_scale", "lora_scale2"]:
                try:
                    if key == "base_model":
                        print(f"Received {key}: {value.decode('utf-8', errors='replace')}")
                    elif key != "command":
                        print(f"Received {key}: {value.decode('utf-8', errors='replace')}")
                except Exception as e:
                    print(f"Error decoding value for {key}: {e}")

            # 处理bypass_mode参数
            if key == "bypass_mode":
                try:
                    bypass_value = value.decode('utf-8').lower()
                    bypass_mode = bypass_value in ["true", "1", "yes", "y"]
                    print(f"设置bypass_mode: {bypass_mode}")
                except Exception as e:
                    print(f"解析bypass_mode参数时出错: {e}")
            
            if key == "run":
                if command_state == 1 and len(command) > 0:
                    print(f"Executing command: {command}")
                    # Restore 'paths' command handling
                    if command == "paths":
                        print(f"设置路径: 基础模型={base_m}, LoRA1={lora_m}, LoRA2={lora_m2}")
                        setModelPaths(base_m, taesd_model, lora_m, lora_m2)
                        # 发送确认消息回客户端
                        try:
                            client_socket.send(b"paths_set")
                            print("Sent confirmation: paths_set")
                        except Exception as e:
                            print(f"Failed to send confirmation: {e}")
                    elif command == "load":
                        print(f"Loading pipeline with: width={width}, height={height}, seed={seed}, strength={strength}")
                        print(f"LoRA1 scale={lora_scale}, LoRA2 scale={lora_scale2}")
                        print(f"Using prompt: '{prompt}'")
                        print(f"Using negative prompt: '{neg_prompt}'")
                        print(f"Delta={delta}, do_add_noise={do_add_noise}")
                        print(f"相似图像过滤: 启用={enable_similar_image_filter}, 阈值={similar_image_filter_threshold}, 最大跳帧={similar_image_filter_max_skip_frame}")
                        print(f"引导尺度: {guidance_scale}")
                        print(f"绕过模式: {bypass_mode}")
                        try:
                            # 如果启用了bypass模式，仍返回成功，但不需要实际加载模型
                            if bypass_mode:
                                print("绕过模式已启用，跳过Pipeline加载")
                                # 如果pipeline_object已存在，保留以便保持API兼容
                                if pipeline_object is None:
                                    # 创建一个最小的Pipeline对象，只是为了保持API兼容
                                    try:
                                        loadPipeline(width, height, seed, use_vae, use_lora, acc_mode, prompt, neg_prompt, 
                                                    strength, lora_scale, lora_scale2, delta, do_add_noise,
                                                    enable_similar_image_filter, similar_image_filter_threshold, 
                                                    similar_image_filter_max_skip_frame, guidance_scale)
                                    except Exception as e:
                                        print(f"尝试在bypass模式下加载Pipeline时出错: {e}")
                                        # 错误发生时不中断流程，因为我们在bypass模式
                                
                                client_socket.send(b"loaded")
                                print("Sent: loaded - bypass mode, no pipeline needed")
                            else:
                                # 正常加载Pipeline
                                loadPipeline(width, height, seed, use_vae, use_lora, acc_mode, prompt, neg_prompt, 
                                          strength, lora_scale, lora_scale2, delta, do_add_noise,
                                          enable_similar_image_filter, similar_image_filter_threshold, 
                                          similar_image_filter_max_skip_frame, guidance_scale)
                                if pipeline_object is None:
                                    client_socket.send(b"failed")
                                    print("Sent: failed - pipeline is None")
                                else:
                                    client_socket.send(b"loaded")
                                    print("Sent: loaded - pipeline loaded successfully")
                        except Exception as e:
                            if bypass_mode:
                                # 在bypass模式下，即使出错也返回成功
                                client_socket.send(b"loaded")
                                print(f"在bypass模式下忽略错误: {e}, 返回loaded")
                            else:
                                print(f"Error loading pipeline: {e}")
                                client_socket.send(b"failed")
                    elif command == "advance":
                        print(f"Advancing pipeline with prompt: '{prompt}'")
                        print(f"绕过模式: {bypass_mode}")
                        
                        # 检查是否有动态传递的强度参数
                        dynamic_strength = strength
                        if dynamic_strength is not None:
                            print(f"Using dynamic strength: {dynamic_strength}")
                            # 如果有传入的强度值，动态更新scheduler
                            if hasattr(pipeline_object.stream.stream, 'scheduler') and hasattr(pipeline_object.stream.stream.scheduler, 'set_timesteps'):
                                print(f"Updating strength to {dynamic_strength} on scheduler")
                                pipeline_object.stream.stream.scheduler.set_timesteps(50, pipeline_object.stream.stream.device, strength=dynamic_strength)
                                
                        # 检查是否有动态传递的LoRA强度
                        if hasattr(pipeline_object.stream.stream, 'lora_scale'):
                            # 更新LoRA字典中的强度值
                            print(f"Updating LoRA scales: LoRA1={lora_scale}, LoRA2={lora_scale2}")
                            pipeline_object.stream.stream.lora_scale = lora_scale
                        
                        # 有些模型在进行每次推理前需要重新设置LoRA权重
                        if hasattr(pipeline_object.stream, 'update_lora_weights') and callable(getattr(pipeline_object.stream, 'update_lora_weights', None)):
                            try:
                                # 更新两个LoRA的权重 - 简化处理，实际实现可能需要调整
                                pipeline_object.stream.update_lora_weights(lora_scale, lora_scale2)
                                print(f"Updated LoRA weights")
                            except Exception as e:
                                print(f"Error updating LoRA weights: {e}")
                        
                        output_bytes = runPipeline(image.getvalue(), prompt)
                        if output_bytes:
                            client_socket.sendall(output_bytes)
                            client_socket.send(b"||||")
                            print(f"Sent generated image, size: {len(output_bytes)} bytes")
                        else:
                            print("Failed to generate image")
                            client_socket.send(b"failed")
                    elif command == "unload":
                        print("Unloading pipeline")
                        pipeline_object = None
                    else:
                        print(f"Unknown command: {command}")
                    command = ""
                    command_state = 0
            elif key == "command":
                try:
                    command = value.decode('utf-8', errors='replace')
                    print(f"Command received: {command}")
                    command_state = 1
                except Exception as e:
                    print(f"Error decoding command: {e}")
                    command = ""
                    command_state = 0
            elif key == "width":
                try:
                    width = int(value)
                except ValueError:
                    print(f"Invalid width value: {value}")
            elif key == "height":
                try:
                    height = int(value)
                except ValueError:
                    print(f"Invalid height value: {value}")
            elif key == "seed":
                try:
                    raw_seed_value = value.decode('utf-8', errors='replace')
                    print(f"Received raw seed value: '{raw_seed_value}'")
                    seed = int(raw_seed_value)
                    print(f"Parsed seed value successfully: {seed}")
                except ValueError as ve:
                    print(f"Invalid seed value: '{value}', error: {ve}")
                    try:
                        # 尝试使用二进制形式解析
                        seed_bytes = bytes(value)
                        print(f"Trying to parse seed from binary: {seed_bytes}")
                        if len(seed_bytes) > 0:
                            try:
                                # 尝试使用第一个整数字节
                                seed = int.from_bytes(seed_bytes[:4], byteorder='little')
                                print(f"Parsed seed from binary: {seed}")
                            except:
                                # 如果二进制转换失败，保留默认值
                                print(f"Binary parsing failed, using default seed: {seed}")
                        else:
                            print(f"Empty seed value, using default: {seed}")
                    except Exception as e:
                        print(f"Error in seed fallback parsing: {e}")
                        # 保留默认种子值
                except Exception as e:
                    print(f"Unexpected error parsing seed: {e}")
                    # 保留默认种子值
            elif key == "strength":
                try:
                    strength_value = float(value)
                    # 更新局部和全局strength值
                    strength = strength_value
                    global default_strength
                    default_strength = strength_value
                    print(f"Setting strength to: {strength} and updating default strength")
                except ValueError:
                    print(f"Invalid strength value: {value}, will use default: {default_strength}")
            elif key == "lora_scale":
                try:
                    lora_scale_value = float(value)
                    if 0 < lora_scale_value <= 1.0:
                        lora_scale = lora_scale_value
                        global default_lora_scale
                        default_lora_scale = lora_scale_value
                        print(f"Setting LoRA1 scale to: {lora_scale} and updating default")
                    else:
                        print(f"LoRA1 scale out of range (0-1): {lora_scale_value}, using default: {default_lora_scale}")
                        lora_scale = default_lora_scale
                except ValueError:
                    print(f"Invalid LoRA1 scale value: {value}, will use default: {default_lora_scale}")
            elif key == "lora_scale2":
                try:
                    lora_scale2_value = float(value)
                    if 0 < lora_scale2_value <= 1.0:
                        lora_scale2 = lora_scale2_value
                        global default_lora_scale2
                        default_lora_scale2 = lora_scale2_value
                        print(f"Setting LoRA2 scale to: {lora_scale2} and updating default")
                    else:
                        print(f"LoRA2 scale out of range (0-1): {lora_scale2_value}, using default: {default_lora_scale2}")
                        lora_scale2 = default_lora_scale2
                except ValueError:
                    print(f"Invalid LoRA2 scale value: {value}, will use default: {default_lora_scale2}")
            elif key == "use_vae":
                try:
                    use_vae = (int(value) > 0)
                except ValueError:
                    print(f"Invalid use_vae value: {value}")
            elif key == "use_lora":
                try:
                    use_lora = (int(value) > 0)
                except ValueError:
                    print(f"Invalid use_lora value: {value}")
            elif key == "acceleration":
                try:
                    acc_mode = value.decode('utf-8', errors='replace')
                except Exception as e:
                    print(f"Error decoding acceleration: {e}")
                    acc_mode = "tensorrt"  # 默认值
            elif key == "prompt":
                try:
                    prompt = value.decode('utf-8', errors='replace')
                except Exception as e:
                    print(f"Error decoding prompt: {e}")
                    prompt = ""  # 如果无法解码，使用空字符串
            elif key == "neg_prompt":
                try:
                    neg_prompt = value.decode('utf-8', errors='replace')
                except Exception as e:
                    print(f"Error decoding negative prompt: {e}")
                    neg_prompt = ""
            elif key == "base_model":
                try:
                    base_m = value.decode('utf-8', errors='replace')
                except Exception as e:
                    print(f"Error decoding base_model: {e}")
            elif key == "lora_model":
                try:
                    lora_m = value.decode('utf-8', errors='replace')
                    if lora_m and os.path.exists(lora_m):
                        print(f"有效的LoRA1模型路径: {lora_m}")
                    else:
                        print(f"LoRA1模型路径无效或不存在: {lora_m}")
                        lora_m = None
                except Exception as e:
                    print(f"解析lora_model时出错: {e}")
                    lora_m = None
            elif key == "lora_model2":
                try:
                    lora_m2 = value.decode('utf-8', errors='replace')
                    if lora_m2 and os.path.exists(lora_m2):
                        print(f"Valid second LoRA model path: {lora_m2}")
                    else:
                        print(f"Second LoRA model path not valid or not found: {lora_m2}")
                        lora_m2 = None
                except Exception as e:
                    print(f"Error decoding lora_model2: {e}")
                    lora_m2 = None
            elif key == "image":
                try:
                    image = io.BytesIO(value)
                    print(f"Received image data, size: {len(value)} bytes")
                except Exception as e:
                    print(f"Error processing image data: {e}")
                    image = io.BytesIO()  # 使用空缓冲区
            elif key == "image_base64":
                try:
                    # 处理Base64编码的图像数据
                    import base64
                    base64_str = value.decode('utf-8', errors='replace')
                    img_data = base64.b64decode(base64_str)
                    image = io.BytesIO(img_data)
                    print(f"接收到Base64图像数据，解码后大小: {len(img_data)} 字节")
                    
                    # 在bypass模式下，不进行颜色空间转换，直接使用原始图像
                    if bypass_mode:
                        print("绕过模式：保持原始图像数据不变")
                    elif is_linear_space:
                        try:
                            # 打开图像
                            temp_image = Image.open(image)
                            # 分析原始图像RGB值
                            arr = np.array(temp_image)
                            print(f"原始图像(线性空间)平均RGB值: R={arr[:,:,0].mean():.2f}, G={arr[:,:,1].mean():.2f}, B={arr[:,:,2].mean():.2f}")
                            
                            # 从线性空间转换到Gamma空间进行存储
                            print(f"进行颜色空间转换: 线性 -> Gamma")
                            arr_float = arr.astype(np.float32) / 255.0
                            arr_gamma = np.power(arr_float, 1/2.2) * 255.0
                            arr_gamma = np.clip(arr_gamma, 0, 255).astype(np.uint8)
                            
                            # 创建Gamma空间图像并存回BytesIO
                            gamma_image = Image.fromarray(arr_gamma)
                            temp_io = io.BytesIO()
                            gamma_image.save(temp_io, format="PNG")
                            image = io.BytesIO(temp_io.getvalue())
                            
                            # 分析转换后的RGB值
                            print(f"转换后(Gamma空间)平均RGB值: R={arr_gamma[:,:,0].mean():.2f}, G={arr_gamma[:,:,1].mean():.2f}, B={arr_gamma[:,:,2].mean():.2f}")
                        except Exception as e:
                            print(f"预处理颜色空间时出错: {e}")
                            import traceback
                            traceback.print_exc()
                except Exception as e:
                    print(f"处理Base64图像数据时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    image = io.BytesIO()  # 使用空缓冲区
            elif key == "delta":
                try:
                    delta_value = float(value)
                    if 0 < delta_value <= 1.0:
                        delta = delta_value
                        print(f"设置Delta为: {delta}")
                    else:
                        print(f"Delta值超出范围(0-1): {delta_value}，使用默认值: {delta}")
                except ValueError:
                    print(f"无效的Delta值: {value}，使用默认值: {delta}")
            elif key == "do_add_noise":
                try:
                    do_add_noise = bool(int(value))
                    print(f"设置添加噪声: {do_add_noise}")
                except ValueError:
                    print(f"无效的添加噪声值: {value}，使用默认值: {do_add_noise}")
            elif key == "enable_similar_filter":
                try:
                    enable_similar_image_filter = bool(int(value))
                    print(f"设置图像相似度过滤: {enable_similar_image_filter}")
                except ValueError:
                    print(f"无效的图像相似度过滤值: {value}，使用默认值: {enable_similar_image_filter}")
            elif key == "similar_threshold":
                try:
                    threshold_value = float(value)
                    if 0 < threshold_value <= 1.0:
                        similar_image_filter_threshold = threshold_value
                        print(f"设置相似度阈值为: {similar_image_filter_threshold}")
                    else:
                        print(f"相似度阈值超出范围(0-1): {threshold_value}，使用默认值: {similar_image_filter_threshold}")
                except ValueError:
                    print(f"无效的相似度阈值: {value}，使用默认值: {similar_image_filter_threshold}")
            elif key == "max_skip_frame":
                try:
                    max_skip = int(value)
                    if max_skip > 0:
                        similar_image_filter_max_skip_frame = max_skip
                        print(f"设置最大跳帧数为: {similar_image_filter_max_skip_frame}")
                    else:
                        print(f"最大跳帧数必须大于0: {max_skip}，使用默认值: {similar_image_filter_max_skip_frame}")
                except ValueError:
                    print(f"无效的最大跳帧数: {value}，使用默认值: {similar_image_filter_max_skip_frame}")
            elif key == "guidance_scale":
                try:
                    guidance_scale_value = float(value)
                    if guidance_scale_value > 0:
                        guidance_scale = guidance_scale_value
                        print(f"设置引导尺度为: {guidance_scale}")
                    else:
                        print(f"引导尺度必须大于0: {guidance_scale_value}，使用默认值: {guidance_scale}")
                except ValueError:
                    print(f"无效的引导尺度值: {value}，使用默认值: {guidance_scale}")
            elif key == "is_linear_space":
                try:
                    is_linear_value = value.decode('utf-8').lower()
                    is_linear_space = is_linear_value in ["true", "1", "yes", "y"]
                    print(f"Unity报告的颜色空间: {'线性空间' if is_linear_space else 'Gamma空间'}")
                except Exception as e:
                    print(f"解析颜色空间参数时出错: {e}")
                    # 默认假设为Gamma空间
                    is_linear_space = False
                    print("默认使用Gamma颜色空间")
            else:
                print(f"Unknown data-buffer key: {key}")

        # 参数解析完成后，进行检查
        if seed == -1:
            print("WARNING: No valid seed received from Unity! Using random seed instead.")
            seed = random.randint(0, 1000000)
            print(f"Generated random seed: {seed}")
        
        # 确保lora_scale有效，防止NoneType比较错误
        if lora_scale is None:
            print(f"WARNING: lora_scale is None, setting to default 0.85")
            lora_scale = 0.85
        elif not isinstance(lora_scale, (int, float)) or lora_scale <= 0 or lora_scale > 1:
            print(f"WARNING: Invalid LoRA scale value: {lora_scale}, resetting to default 0.85")
            lora_scale = 0.85
            
        print(f"Final parameters after parsing: Width={width}, Height={height}, Seed={seed}, Strength={strength}, LoRA Scale={lora_scale}")
            
    except Exception as e:
        print(f"Error processing command: {e}")
        import traceback
        traceback.print_exc()
        try:
            client_socket.send(b"badreq")
        except:
            pass

def receiveCompleteData(client_socket):
    """安全地接收完整的数据包，处理网络错误和编码问题"""
    start_marker = b"|start|"
    end_marker = b"|end|"
    data_buffer = b""
    max_buffer_size = 10 * 1024 * 1024  # 10MB最大缓冲区限制，防止内存溢出
    
    while True:
        try:
            chunk = client_socket.recv(4096)
            if not chunk:
                print("Connection closed by client - no data received")
                return b"", b""  # Connection closed
                
            # 记录收到的数据大小
            print(f"Received chunk: {len(chunk)} bytes")
            
            # 添加到缓冲区
            data_buffer += chunk
            
            # 检查缓冲区大小，防止内存溢出
            if len(data_buffer) > max_buffer_size:
                print(f"WARNING: Buffer exceeded max size ({max_buffer_size} bytes), truncating")
                # 只保留最后1MB的数据，丢弃前面的部分
                data_buffer = data_buffer[-1024*1024:]
            
            # 检查是否找到完整消息
            start_idx = data_buffer.find(start_marker)
            end_idx = data_buffer.find(end_marker)
            
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                # 发现完整消息
                complete_data = data_buffer[start_idx : end_idx + len(end_marker)]
                remaining_data = data_buffer[end_idx + len(end_marker):]
                print(f"Found complete message: {len(complete_data)} bytes, remaining: {len(remaining_data)} bytes")
                return complete_data, remaining_data
                
            # 没有完整消息，继续接收数据
            print(f"Incomplete message in buffer ({len(data_buffer)} bytes), continuing to receive...")
                
        except socket.timeout:
            print("Socket timeout while receiving data")
            return b"", data_buffer  # 返回已接收的数据
        except socket.error as e:
            print(f"Socket error: {e}")
            return b"", b""  # 网络错误，返回空
        except Exception as e:
            print(f"Unexpected error in receiveCompleteData: {e}")
            import traceback
            traceback.print_exc()
            return b"", b""  # 其他错误，返回空

def startTcpServer(host='127.0.0.1', port=9999):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Allow address reuse
    
    # 设置更长的超时时间
    server_socket.settimeout(60.0)
    
    # 增加接收和发送缓冲区大小
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)  # 256KB接收缓冲区
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)  # 256KB发送缓冲区
    
    try:
        server_socket.bind((host, port))
        server_socket.listen(1) # Listen for 1 connection
        print(f"TCP Server started on {host}:{port}")
        
        while True:
            print("Waiting for a client connection...")
            try:
                client_socket, client_address = server_socket.accept()
                print(f"Client connected: {client_address}")
                
                # 设置客户端连接参数
                client_socket.settimeout(60.0)  # 增加超时时间到60秒
                client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # 禁用Nagle算法
                
                # 增加接收和发送缓冲区大小
                client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)
                client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)
                
                remaining_data = b""
                try:
                    while True:
                        complete_data, remaining_data = receiveCompleteData(client_socket)
                        if complete_data:
                            #print(f"Received complete data: {complete_data[:100]}...") # Optional: Log received data
                            processData(client_socket, complete_data)
                        elif not remaining_data and not client_socket.fileno() == -1: # No complete data, buffer empty, check if socket alive
                            # If receiveCompleteData returned empty due to timeout or partial read, keep waiting
                            pass
                        else:
                            # Connection likely closed or error occurred in receiveCompleteData
                            print("No complete data received and connection may be closed.")
                            break
                
                except ConnectionResetError:
                    print(f"Client {client_address} disconnected unexpectedly.")
                except socket.timeout:
                    print(f"Client {client_address} timed out.")
                except Exception as e:
                    print(f"Error handling client {client_address}: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    print(f"Closing connection to {client_address}")
                    try:
                        client_socket.close()
                    except:
                        pass # Ignore errors if already closed
            except socket.timeout:
                print("Server socket timed out waiting for connection, continuing...")
            except Exception as e:
                print(f"Error accepting connection: {e}")
                import traceback
                traceback.print_exc()
    except KeyboardInterrupt:
        print("Server is shutting down due to keyboard interrupt.")
    except Exception as e:
        print(f"Fatal server error: {e}")
        import traceback
        traceback.print_exc()
        raise  # 重新抛出异常，触发自动重启
    finally:
        print("Closing server socket.")
        try:
            server_socket.close()
        except:
            pass

if __name__ == "__main__":
    print("Starting image predictor...")
    # 添加自动重启功能
    max_restarts = 5
    restart_count = 0
    
    while restart_count < max_restarts:
        try:
            print(f"Server start attempt #{restart_count+1}")
            startTcpServer()
        except KeyboardInterrupt:
            print("Server shutdown requested via keyboard interrupt.")
            break
        except Exception as e:
            restart_count += 1
            print(f"Server crashed with error: {e}")
            import traceback
            traceback.print_exc()
            print(f"Restarting server in 5 seconds... (attempt {restart_count}/{max_restarts})")
            import time
            time.sleep(5)
    
    print("Server shutdown complete.")
