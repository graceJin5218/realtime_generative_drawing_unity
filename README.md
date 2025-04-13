# streamdiffusion_unity
A unity interaction project with the StreamDiffusion backend

### How to install

1. Enable Git LFS:
<em>git lfs install</em>

2. Clone this repository:
<em>git clone --recurse-submodules https://github.com/code-2-art/streamdiffusion_unity.git</em>

You may also update submodules later:
<em>git clone https://github.com/code-2-art/streamdiffusion_unity.git</em>
<em>git submodule update --init</em>

3. Configure streamdiffusion (submodule in StreamAssets/streamdiffusion) as introduced in:
https://github.com/cumulo-autumn/StreamDiffusion/blob/main/README.md#installation

4. Copy envrionment folder (e.g., <anaconda>/envs/streamdiffusion) to StreamAssets/envs/streamdiffusion

5. Open SampleScene.unity and enjoy the risk on your own!
When "Acceleration" is "tensorrt", the system will halt for a long time to generate native engine data. Please be patient.
For Graphics cards with low memory, change "Acceleration" from "tensorrt" to "xformers" in Unity Inspector
