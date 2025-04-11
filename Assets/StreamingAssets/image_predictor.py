import sys, os, io, socket, torch
from pydantic import BaseModel, Field
from PIL import Image
from typing import Literal, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "streamdiffusion"))
from utils.wrapper import StreamDiffusionWrapper

base_model = "./models/kohaku-v2.1"
taesd_model = "./models/taesd"
lora_model = "./models/lcm-lora-sdv1-5"
pipeline_object = None

class Pipeline:
    def __init__(self, w: int, h: int, seed: int, device: torch.device, torch_dtype: torch.dtype,
                 use_vae: bool, use_lora: bool, gc_mode: Literal["img2img", "txt2img"], acc_mode: str,
                 positive_prompt: str, negative_prompt: str = ""):
        self.stream = StreamDiffusionWrapper(
            model_id_or_path=base_model,
            vae_id=taesd_model,
            lcm_lora_id=lora_model,
            use_tiny_vae=use_vae,
            use_lcm_lora=use_lora,
            use_denoising_batch=True,
            device=device,
            dtype=torch_dtype,
            t_index_list=[35, 45],
            frame_buffer_size=1,
            width=w, height=h,
            output_type="pil",
            warmup=10,
            acceleration=acc_mode,
            mode=gc_mode,
            seed=seed,
            cfg_type="none",
            use_safety_checker=False,
            # enable_similar_image_filter=True,
            # similar_image_filter_threshold=0.98,
            engine_dir="engines",
        )

        self.stream.prepare(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            guidance_scale=1.2, delta=0.5
        )

    def predict(self, input: Image.Image, new_prompt: Optional[str] = None) -> Image.Image:
        image_tensor = self.stream.preprocess_image(input)
        for _ in range(self.stream.batch_size - 1):
            self.stream(image=image_tensor, prompt=new_prompt)
        output = self.stream(image=image_tensor, prompt=new_prompt)
        return output

def setModelPaths(base_m: str, tiny_vae_m: str, lcm_lora_m: str):
    global base_model
    global taesd_model
    global lora_model
    base_model = base_m
    taesd_model = tiny_vae_m
    lora_model = lcm_lora_m

def loadPipeline(w: int, h: int, seed: int, use_vae: bool, use_lora: bool,
                 acc_mode: str, positive_prompt: str, negative_prompt: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16
    global pipeline_object
    pipeline_object = Pipeline(w, h, seed, device, torch_dtype, use_vae, use_lora, "img2img",
                               acc_mode, positive_prompt, negative_prompt)

def runPipeline(input_bytes, new_prompt: str):
    output_io = io.BytesIO()
    if pipeline_object is None:
        return None
    else:
        input_image = Image.open(io.BytesIO(input_bytes))
        output_image = pipeline_object.predict(input_image, new_prompt);
        output_image.save(output_io, format="JPEG")
        #output_image.save("./output_u3d.jpg")
        return output_io.getvalue()

def processData(client_socket, data):
    global pipeline_object
    data = data.strip(b"|start|").strip(b"|end|")
    parts = data.split(b"||")

    use_vae = True
    use_lora = True
    width = 512
    height = 512
    seed = 2
    command_state = 0
    command = ""
    acc_mode = "tensorrt"
    prompt = ""
    neg_prompt = ""
    base_m = base_model
    taesd_m = taesd_model
    lora_m = lora_model
    image = io.BytesIO()

    try:
        for i in range(0, len(parts), 2):
            if i + 1 >= len(parts):
                break
            key = parts[i].decode('utf-8')
            value = parts[i + 1]
            
            if key == "run":
                if command_state == 1 and len(command) > 0:
                    if command == "paths":
                        setModelPaths(base_m, taesd_m, lora_m)
                    elif command == "load":
                        loadPipeline(width, height, seed, use_vae, use_lora, acc_mode, prompt, neg_prompt)
                        if pipeline_object is None:
                            client_socket.send(b"failed")
                        else:
                            client_socket.send(b"loaded")
                    elif command == "advance":
                        output_bytes = runPipeline(image.getvalue(), prompt)
                        #print(f"Advanced: {len(output_bytes)}")
                        client_socket.sendall(output_bytes)
                        client_socket.send(b"||||")
                    elif command == "unload":
                        pipeline_object = None
                    else:
                        print(f"Unknown command: {command}")
                    command = ""
                    command_state = 0
            elif key == "command":
                command = value.decode('utf-8')
                command_state = 1
            elif key == "width":
                width = int(value)
            elif key == "height":
                height = int(value)
            elif key == "seed":
                seed = int(value)
            elif key == "use_vae":
                use_vae = (int(value) > 0)
            elif key == "use_lora":
                use_lora = (int(value) > 0)
            elif key == "acceleration":
                acc_mode = value.decode('utf-8')
            elif key == "prompt":
                prompt = value.decode('utf-8')
            elif key == "neg_prompt":
                neg_prompt = value.decode('utf-8')
            elif key == "base_model":
                base_m = value.decode('utf-8')
            elif key == "taesd_model":
                taesd_m = value.decode('utf-8')
            elif key == "lora_model":
                lora_m = value.decode('utf-8')
            elif key == "image":
                image = io.BytesIO(value)
            else:
                print(f"Unknown data-buffer key: {key}")
    except Exception as e:
        client_socket.send(b"badreq")


def receiveCompleteData(client_socket):
    start_marker = b"|start|"
    end_marker = b"|end|"
    data_buffer = b""
    while True:
        chunk = client_socket.recv(4096)
        if not chunk:
            break
        data_buffer += chunk
        if data_buffer.startswith(start_marker) and end_marker in data_buffer:
            end_index = data_buffer.find(end_marker) + len(end_marker)
            complete_data = data_buffer[:end_index]
            remaining_data = data_buffer[end_index:]
            return complete_data, remaining_data
    return b"", b""

def startTcpServer(host='127.0.0.1', port=9999):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"TCP Server started on {host}:{port}")
    try:
        while True:
            client_socket, client_address = server_socket.accept()
            remaining_data = b""
            try:
                while True:
                    if remaining_data:
                        complete_data, remaining_data = receiveCompleteData(client_socket)
                        if complete_data:
                            processData(client_socket, complete_data)
                    else:
                        complete_data, remaining_data = receiveCompleteData(client_socket)
                        if complete_data:
                            processData(client_socket, complete_data)
            except Exception as e:
                print(f"Parsing data error: {e}")
            finally:
                client_socket.close()
    except KeyboardInterrupt:
        print("Server is shutting down.")
    finally:
        server_socket.close()
        print("Server socket closed.")

if __name__ == "__main__":
    print("Starting...")
    startTcpServer()