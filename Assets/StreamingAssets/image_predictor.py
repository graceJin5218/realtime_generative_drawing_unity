import sys, os, io, socket, torch
from pydantic import BaseModel, Field
from PIL import Image
from typing import Literal, Optional
import numpy as np
import random

sys.path.append(os.path.join(os.path.dirname(__file__), "streamdiffusion"))
from utils.wrapper import StreamDiffusionWrapper

# 获取当前脚本所在目录的绝对路径
base_dir = os.path.dirname(os.path.abspath(__file__))

# 尝试在多个位置查找模型文件
model_filename = "photonLCM_v10.safetensors"
possible_paths = [
    os.path.join(base_dir, "models", "Model", model_filename),  # 标准路径
    os.path.join("D:", "streamdiffusion_unity", "Assets", "StreamingAssets", "models", "Model", model_filename),  # 绝对路径
]

# 检查文件存在于哪个路径
found_model_path = None
for path in possible_paths:
    if os.path.exists(path):
        found_model_path = path
        print(f"Found model file at: {found_model_path}")
        break

if found_model_path is None:
    # 如果找不到，使用默认路径并发出警告
    found_model_path = os.path.join(base_dir, "models", "Model", model_filename)
    print(f"WARNING: Could not find model file at any expected location. Using default path: {found_model_path}")

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
lora_filename = "tbh123-.safetensors"
lora_possible_paths = [
    os.path.join(base_dir, "models", "LoRA", lora_filename),  # 标准路径
    os.path.join("D:", "streamdiffusion_unity", "Assets", "StreamingAssets", "models", "LoRA", lora_filename),  # 绝对路径
]

# 检查LoRA文件存在于哪个路径
found_lora_path = None
for path in lora_possible_paths:
    if os.path.exists(path):
        found_lora_path = path
        print(f"Found LoRA file at: {found_lora_path}")
        break

if found_lora_path is None:
    # 如果找不到LoRA，暂时设为None
    found_lora_path = None
    print(f"WARNING: Could not find LoRA file. LoRA will not be used.")

# 使用找到的路径
base_model = found_model_path
taesd_model = found_vae_path  # 使用找到的VAE路径
lora_model = found_lora_path  # 使用找到的LoRA文件路径或None
pipeline_object = None
default_strength = 2.0  # 默认strength参数值

print(f"Initial model paths set to:\nBase={base_model}\nVAE={taesd_model}\nLoRA={lora_model}")
print(f"Default strength parameter: {default_strength}")

class Pipeline:
    def __init__(self, w: int, h: int, seed: int, device: torch.device, torch_dtype: torch.dtype,
                 use_vae: bool, use_lora: bool, gc_mode: Literal["img2img", "txt2img"], acc_mode: str,
                 positive_prompt: str, negative_prompt: str = "", preloaded_pipe=None, model_path=None, lora_dict=None):
        # 使用传入的特定路径
        actual_model_path = model_path if model_path else base_model
        print(f"Initializing pipeline with model path: {actual_model_path}")
        
        # 获取全局VAE模型路径
        global taesd_model
        local_taesd_model = taesd_model
        
        # 打印VAE信息
        is_local_vae = False
        is_vae_folder = False
        
        if isinstance(local_taesd_model, str):
            if local_taesd_model != "madebyollin/taesd" and os.path.exists(local_taesd_model):
                if os.path.isdir(local_taesd_model) and os.path.exists(os.path.join(local_taesd_model, "config.json")):
                    is_local_vae = True
                    is_vae_folder = True
                    print(f"Using local VAE folder: {local_taesd_model}")
                elif local_taesd_model.endswith(".safetensors"):
                    # 如果是safetensors文件，检查同一目录下是否有config.json
                    vae_dir = os.path.dirname(local_taesd_model)
                    if os.path.exists(os.path.join(vae_dir, "config.json")):
                        local_taesd_model = vae_dir
                        is_local_vae = True
                        is_vae_folder = True
                        print(f"Found config.json in same directory as safetensors, using folder: {local_taesd_model}")
                    else:
                        print(f"WARNING: Local VAE file {local_taesd_model} is a safetensors file without config.json, switching to HuggingFace model")
                        local_taesd_model = "madebyollin/taesd"
                        is_local_vae = False
            else:
                print(f"Using HuggingFace VAE model: {local_taesd_model}")
        
        # 决定是否使用LoRA
        use_lcm_lora_value = False
        if use_lora and lora_model is not None and os.path.exists(lora_model):
            print(f"LoRA will be used: {lora_model}")
            use_lcm_lora_value = True
            
        # 使用传入的LoRA字典
        lora_dict_to_use = lora_dict if lora_dict is not None else ({lora_model: 0.85} if use_lora and lora_model is not None and os.path.exists(lora_model) else None)
        if lora_dict_to_use:
            print(f"Using LoRA with dictionary: {lora_dict_to_use}")
        
        if preloaded_pipe is not None:
            print("Using preloaded Stable Diffusion Pipeline")
            self.stream = StreamDiffusionWrapper(
                pipe=preloaded_pipe,
                vae_id=local_taesd_model,
                lcm_lora_id=lora_model if use_lcm_lora_value else None,
                lora_dict=lora_dict_to_use,  # 使用传入的LoRA字典
                use_tiny_vae=use_vae,
                use_lcm_lora=use_lcm_lora_value,
                use_denoising_batch=True,
                device=device,
                dtype=torch_dtype,
                t_index_list=[15, 25],
                frame_buffer_size=1,
                width=w, height=h,
                output_type="tensor",  # 修改为tensor类型输出
                warmup=10,
                acceleration=acc_mode,
                mode=gc_mode,
                seed=seed,
                cfg_type="none",
                use_safety_checker=False,
                engine_dir="engines",
            )
        else:
            print("Loading model from path")
            self.stream = StreamDiffusionWrapper(
                model_id_or_path=actual_model_path,
                vae_id=local_taesd_model,
                lcm_lora_id=lora_model if use_lcm_lora_value else None,
                lora_dict=lora_dict_to_use,  # 使用传入的LoRA字典
                use_tiny_vae=use_vae,
                use_lcm_lora=use_lcm_lora_value,
                use_denoising_batch=True,
                device=device,
                dtype=torch_dtype,
                t_index_list=[15, 25],
                frame_buffer_size=1,
                width=w, height=h,
                output_type="tensor",  # 修改为tensor类型输出
                warmup=10,
                acceleration=acc_mode,
                mode=gc_mode,
                seed=seed,
                cfg_type="none",
                use_safety_checker=False,
                engine_dir="engines",
            )
            
        # 初始化阶段不调用prepare，将在loadPipeline函数中调用

    def predict(self, input: Image.Image, new_prompt: Optional[str] = None) -> torch.Tensor:
        try:
            # 预处理输入图像为张量
            print(f"Input image size: {input.size}")
            image_tensor = self.stream.preprocess_image(input)
            
            # 使用新提示词（如果有）
            if new_prompt:
                print(f"Using prompt: {new_prompt}")
            
            # 如果批处理大小大于1，先运行几次
            for _ in range(self.stream.batch_size - 1):
                self.stream(image=image_tensor, prompt=new_prompt)
            
            # 获取最终输出
            output = self.stream(image=image_tensor, prompt=new_prompt)
            print(f"Output type: {type(output)}, ", end="")
            if isinstance(output, torch.Tensor):
                print(f"shape: {output.shape}")
            else:
                print(f"PIL image size: {output.size}")
            
            return output
        except Exception as e:
            print(f"Error in predict: {e}")
            import traceback
            traceback.print_exc()
            # 返回一个空的张量作为错误指示
            return torch.zeros((3, 512, 512), dtype=torch.float32)

# Re-add setModelPaths function
def setModelPaths(base_m: str, tiny_vae_m: str, lcm_lora_m: str):
    global base_model
    global taesd_model
    global lora_model
    
    print(f"Received paths: Base={base_m}, VAE={tiny_vae_m}, LoRA={lcm_lora_m}")
    
    # 更新基础模型路径
    if base_m and os.path.exists(base_m):
        base_model = base_m
        print(f"Base model path updated to: {base_model}")
    
    # 更新VAE模型路径
    if tiny_vae_m:
        if os.path.exists(tiny_vae_m) and os.path.exists(os.path.join(tiny_vae_m, "config.json")):
            taesd_model = tiny_vae_m
            print(f"VAE model path updated to local folder: {taesd_model}")
        elif tiny_vae_m.startswith("madebyollin"):
            taesd_model = tiny_vae_m
            print(f"VAE model set to HuggingFace model: {taesd_model}")
        else:
            print(f"VAE path not valid or missing config.json: {tiny_vae_m}")
            config_exists = False
            # 检查是否有效的本地VAE文件夹路径
            for possible_path in [tiny_vae_m, os.path.dirname(tiny_vae_m)]:
                config_path = os.path.join(possible_path, "config.json")
                if os.path.exists(config_path):
                    taesd_model = possible_path
                    print(f"Found config.json in directory: {taesd_model}")
                    config_exists = True
                    break
            
            if not config_exists:
                print("Switching to HuggingFace model")
                taesd_model = "madebyollin/taesd"
                print(f"Now using HuggingFace VAE model: {taesd_model}")
    
    # 检查LoRA模型路径
    if lcm_lora_m and os.path.exists(lcm_lora_m):
        lora_model = lcm_lora_m
        print(f"LoRA model path updated to: {lora_model}")
    else:
        print(f"LoRA model path not found or not provided, current: {lora_model}")
    
    print(f"Final paths: Base={base_model}, VAE={taesd_model}, LoRA={lora_model}")

def loadPipeline(w: int, h: int, seed: int, use_vae: bool, use_lora: bool,
                 acc_mode: str, positive_prompt: str, negative_prompt: str, strength: float = None, lora_scale: float = None):
    # 使用全局默认值或传入的值
    global default_strength
    if strength is None:
        strength = default_strength
        print(f"Using default strength value: {strength}")
    else:
        # 如果提供了新的strength值，更新全局默认值
        default_strength = strength
        print(f"Updating default strength to: {default_strength}")
    
    # 处理LoRA强度值
    if lora_scale is None or lora_scale <= 0:
        lora_scale = 0.85  # 默认值
        print(f"Using default LoRA scale: {lora_scale}")
    else:
        print(f"Using provided LoRA scale: {lora_scale}")
    
    # 确保提示词为空时使用空字符串，而不是预设值
    if positive_prompt is None:
        positive_prompt = ""
    if negative_prompt is None:
        negative_prompt = ""
        
    print(f"Loading pipeline with model path: {base_model}")
    print(f"Base model exists: {os.path.exists(base_model)}")
    print(f"VAE model: {taesd_model}")
    print(f"LoRA model: {lora_model}, exists: {os.path.exists(lora_model) if lora_model else False}")
    print(f"Using VAE: {use_vae}, Using LoRA: {use_lora}, LoRA Scale: {lora_scale}")
    print(f"Strength: {strength}")
    print(f"Prompt: '{positive_prompt}'")
    print(f"Negative prompt: '{negative_prompt}'")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16
    global pipeline_object
    
    try:
        print("Initializing pipeline with standard method")
        
        # 决定是否使用LoRA
        use_lcm_lora_value = False
        if use_lora and lora_model is not None and os.path.exists(lora_model):
            print(f"LoRA will be used: {lora_model}")
            use_lcm_lora_value = True
            
        # 创建LoRA字典，使用传入的lora_scale值
        lora_dict_to_use = {lora_model: lora_scale} if use_lora and lora_model is not None and os.path.exists(lora_model) else None
        if lora_dict_to_use:
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
            lora_dict=lora_dict_to_use  # 传递LoRA字典
        )
        
        # 在Pipeline初始化后设置strength参数
        pipeline_object.stream.prepare(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            guidance_scale=1.2, 
            delta=0.5,
        )
        
        # 单独设置strength参数
        if hasattr(pipeline_object.stream.stream, 'scheduler') and hasattr(pipeline_object.stream.stream.scheduler, 'set_timesteps'):
            print(f"Setting strength={strength} on scheduler")
            pipeline_object.stream.stream.scheduler.set_timesteps(50, pipeline_object.stream.stream.device, strength=strength)
    except Exception as e:
        print(f"ERROR IN LOADING PIPELINE: {e}")
        import traceback
        traceback.print_exc()
        pipeline_object = None

def runPipeline(input_bytes, new_prompt: str):
    output_io = io.BytesIO()
    if pipeline_object is None:
        return None
    else:
        try:
            # 确保提示词不为None
            if new_prompt is None:
                new_prompt = ""
                
            input_image = Image.open(io.BytesIO(input_bytes))
            output_image = pipeline_object.predict(input_image, new_prompt)
            
            # 检查输出类型并适当转换
            if isinstance(output_image, torch.Tensor):
                print(f"Converting tensor output to PIL image, shape: {output_image.shape}")
                # 如果是张量，转换为PIL图像
                if output_image.dim() == 4:  # [batch, channel, height, width]
                    output_image = output_image.squeeze(0)  # 去掉批处理维度
                
                # 确保输出是在0-1范围内并且是CPU张量
                output_image = output_image.clamp(0, 1).cpu().float()
                
                # 转换为PIL图像
                if output_image.shape[0] == 3:  # [channel, height, width]
                    output_image = output_image.permute(1, 2, 0).numpy()  # [height, width, channel]
                    output_image = (output_image * 255).astype(np.uint8)
                    output_image = Image.fromarray(output_image)
                else:
                    print(f"Unexpected tensor shape: {output_image.shape}")
                    return None
            
            # 保存为JPEG
            output_image.save(output_io, format="JPEG")
            return output_io.getvalue()
        except Exception as e:
            print(f"Error in runPipeline: {e}")
            import traceback
            traceback.print_exc()
            return None

def processData(client_socket, data):
    global pipeline_object
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
        lora_scale = 0.85  # 添加LoRA强度参数，默认为main.py中使用的0.85
        command_state = 0
        command = ""
        acc_mode = "tensorrt"
        prompt = ""
        neg_prompt = ""
        base_m = base_model
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
            if key in ["command", "base_model", "taesd_model", "lora_model", "prompt", "acceleration", "strength"]:
                try:
                    if key == "base_model":
                        print(f"Received {key}: {value.decode('utf-8', errors='replace')}")
                    elif key != "command":
                        print(f"Received {key}: {value.decode('utf-8', errors='replace')}")
                except Exception as e:
                    print(f"Error decoding value for {key}: {e}")

            if key == "run":
                if command_state == 1 and len(command) > 0:
                    print(f"Executing command: {command}")
                    # Restore 'paths' command handling
                    if command == "paths":
                        print(f"Setting paths: base_model={base_m}")
                        setModelPaths(base_m, taesd_model, lora_model)
                        # 发送确认消息回客户端
                        try:
                            client_socket.send(b"paths_set")
                            print("Sent confirmation: paths_set")
                        except Exception as e:
                            print(f"Failed to send confirmation: {e}")
                    elif command == "load":
                        print(f"Loading pipeline with: width={width}, height={height}, seed={seed}, strength={strength}, lora_scale={lora_scale}")
                        print(f"Using prompt: '{prompt}'")
                        print(f"Using negative prompt: '{neg_prompt}'")
                        try:
                            # 使用接收到的lora_scale值创建字典
                            loadPipeline(width, height, seed, use_vae, use_lora, acc_mode, prompt, neg_prompt, strength, lora_scale)
                            if pipeline_object is None:
                                client_socket.send(b"failed")
                                print("Sent: failed - pipeline is None")
                            else:
                                client_socket.send(b"loaded")
                                print("Sent: loaded - pipeline loaded successfully")
                        except Exception as e:
                            print(f"Error loading pipeline: {e}")
                            client_socket.send(b"failed")
                            print("Sent: failed - error loading pipeline")
                    elif command == "advance":
                        print(f"Advancing pipeline with prompt: '{prompt}'")
                        
                        # 检查是否有动态传递的强度参数
                        dynamic_strength = strength
                        if dynamic_strength is not None:
                            print(f"Using dynamic strength: {dynamic_strength}")
                            # 如果有传入的强度值，动态更新scheduler
                            if hasattr(pipeline_object.stream.stream, 'scheduler') and hasattr(pipeline_object.stream.stream.scheduler, 'set_timesteps'):
                                print(f"Updating strength to {dynamic_strength} on scheduler")
                                pipeline_object.stream.stream.scheduler.set_timesteps(50, pipeline_object.stream.stream.device, strength=dynamic_strength)
                                
                        # 检查是否有动态传递的LoRA强度
                        if lora_scale is not None and lora_scale > 0 and lora_scale <= 1.0:
                            # 动态更新LoRA字典中的强度值
                            if hasattr(pipeline_object.stream.stream, 'lora_scale'):
                                print(f"Updating LoRA scale to {lora_scale}")
                                pipeline_object.stream.stream.lora_scale = lora_scale
                            # 有些模型在进行每次推理前需要重新设置LoRA权重
                            if hasattr(pipeline_object.stream, 'update_lora_weights') and callable(getattr(pipeline_object.stream, 'update_lora_weights', None)):
                                try:
                                    pipeline_object.stream.update_lora_weights(lora_scale)
                                    print(f"Updated LoRA weights with scale: {lora_scale}")
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
                    print(f"Received Base64 image data, decoded size: {len(img_data)} bytes")
                except Exception as e:
                    print(f"Error processing Base64 image data: {e}")
                    import traceback
                    traceback.print_exc()
                    image = io.BytesIO()  # 使用空缓冲区
            elif key == "lora_scale":
                try:
                    raw_lora_scale = value.decode('utf-8', errors='replace')
                    print(f"Received raw LoRA scale value: '{raw_lora_scale}'")
                    lora_scale = float(raw_lora_scale)
                    print(f"Parsed LoRA scale value successfully: {lora_scale}")
                except ValueError as ve:
                    print(f"Invalid LoRA scale value: '{value}', error: {ve}, using default: {lora_scale}")
                except Exception as e:
                    print(f"Error processing LoRA scale: {e}, using default: {lora_scale}")
            else:
                print(f"Unknown data-buffer key: {key}")

        # 参数解析完成后，进行检查
        if seed == -1:
            print("WARNING: No valid seed received from Unity! Using random seed instead.")
            seed = random.randint(0, 1000000)
            print(f"Generated random seed: {seed}")
        
        if lora_scale <= 0 or lora_scale > 1:
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