# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment Setup

### Prerequisites
- Python 3.10 with PyTorch 2.5.1+cu124
- CUDA 12.6 support required
- 10GB VRAM minimum (shape generation), 21GB for texture generation, 29GB for full pipeline

### Installation Commands
```bash
# Install PyTorch with CUDA support
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install main dependencies
pip install -r requirements.txt

# Build custom rasterizer
cd hy3dpaint/custom_rasterizer
pip install -e .
cd ../..

# Compile differentiable renderer
cd hy3dpaint/DifferentiableRenderer
bash compile_mesh_painter.sh
cd ../..

# Download RealESRGAN weights for texture generation
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt
```

### Common Runtime Commands
```bash
# Basic demo (shape + texture generation)
python demo.py

# Launch Gradio web interface
python gradio_app.py --model_path tencent/Hunyuan3D-2.1 --subfolder hunyuan3d-dit-v2-1 --texgen_model_path tencent/Hunyuan3D-2.1 --low_vram_mode

# Run via Cog (Docker-based inference)
cog predict -i image=@assets/demo.png

# Training shape generation model
cd hy3dshape
python main.py --config configs/hunyuan3ddit-full-params-finetuning-flowmatching-dinog518-bf16-lr1e5-512.yaml

# Training texture generation model
cd hy3dpaint
python train.py --config cfgs/hunyuan-paint-pbr.yaml
```

## Repository Architecture

This is a **two-stage 3D asset generation system**:

### Stage 1: Shape Generation (`hy3dshape/`)
- **Pipeline**: `Hunyuan3DDiTFlowMatchingPipeline` - Image/text to 3D mesh generation
- **Model**: 3.3B parameter DiT (Diffusion Transformer) with flow matching
- **Input**: Images or text prompts
- **Output**: Untextured 3D mesh (.glb format)
- **Key Files**:
  - `hy3dshape/pipelines.py` - Main inference pipeline
  - `hy3dshape/models/denoisers/hunyuan3ddit.py` - Core DiT model
  - `hy3dshape/models/autoencoders/` - VAE encoder/decoder for 3D shapes

### Stage 2: Texture Generation (`hy3dpaint/`)
- **Pipeline**: `Hunyuan3DPaintPipeline` - PBR texture synthesis for 3D meshes
- **Model**: 2B parameter 2.5D UNet with multiview consistency
- **Input**: 3D mesh + reference image
- **Output**: Textured mesh with PBR materials (albedo, metallic, roughness, normal maps)
- **Key Files**:
  - `textureGenPipeline.py` - Main texture generation pipeline
  - `hunyuanpaintpbr/pipeline.py` - PBR-specific diffusion pipeline
  - `DifferentiableRenderer/` - Custom mesh rendering for texture synthesis

### Integration Points
- Shape generation outputs `.glb` files consumed by texture generation
- Both stages can run independently or together via `demo.py` or `gradio_app.py`

## Key Development Patterns

### Model Loading
```python
# Shape generation
sys.path.insert(0, './hy3dshape')
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')

# Texture generation  
sys.path.insert(0, './hy3dpaint')
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
config = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
paint_pipeline = Hunyuan3DPaintPipeline(config)
```

### Configuration Management
- Shape generation: YAML configs in `hy3dshape/configs/`
- Texture generation: YAML configs in `hy3dpaint/cfgs/`
- Configuration classes: `Hunyuan3DPaintConfig` for texture generation

### GPU Memory Management
- Both pipelines implement GPU memory cleanup via `torch.cuda.empty_cache()`
- Use `low_vram_mode` flag in Gradio app for memory-constrained environments
- Custom CUDA kernels in `hy3dpaint/custom_rasterizer/` require proper compilation

## Important Dependencies and Compatibility

### Torchvision Compatibility Issue
The codebase includes a torchvision compatibility fix:
```python
from torchvision_fix import apply_fix
apply_fix()
```
This addresses missing `torchvision.transforms.functional_tensor` in newer versions.

### Custom CUDA Components
- `custom_rasterizer` - Custom CUDA kernel for efficient texture rasterization
- `mesh_inpaint_processor.so` - Compiled C++/CUDA mesh processing (committed binary)
- Both require CUDA 12.6 and proper compilation environment

### PyMeshLab Dependency
Known issue with PyMeshLab Qt5 symbol conflicts. If experiencing crashes:
- Ensure proper Qt5 installation
- Consider using Docker/Cog environment for stable deployment

## Training Configuration

### Shape Generation Training
- Uses DeepSpeed for distributed training: `hy3dshape/scripts/train_deepspeed.sh`
- Flow matching training with velocity prediction
- Supports LoRA/PEFT fine-tuning via `hy3dshape/utils/trainings/peft.py`

### Texture Generation Training  
- Multi-view diffusion training with position-aware attention
- PBR material property prediction
- Custom data format documented in `hy3dpaint/src/data/pbr_data_format.txt`

## Testing and Validation

### Mini Test Sets
- Shape generation: `hy3dshape/tools/mini_testset/`
- Texture generation: `hy3dpaint/train_examples/`

### Evaluation Metrics
- Shape generation: ULIP-T, ULIP-I, Uni3D-T, Uni3D-I scores
- Texture generation: CLIP-FiD, CMMD, CLIP-I, LPIPS scores

## Known Issues and Limitations

1. **Memory Requirements**: Full pipeline needs 29GB VRAM
2. **PyMeshLab Compatibility**: Qt5 symbol conflicts on some systems  
3. **CUDA Dependency**: Custom kernels require exact CUDA 12.6 version
4. **Torchvision Version**: Needs compatibility fix for versions 0.20+

## Model Configuration Files

Key configuration files that control model behavior:
- `hy3dshape/configs/hunyuan3ddit-*-flowmatching-*.yaml` - Shape generation configs
- `hy3dpaint/cfgs/hunyuan-paint-pbr.yaml` - PBR texture generation config
- `cog.yaml` - Cog/Docker deployment configuration