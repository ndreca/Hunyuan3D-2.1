import os
import shutil
import subprocess
import sys
import time
import traceback

try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")

sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

from PIL import Image
from torch import cuda, Generator
from cog import BasePredictor, BaseModel, Input, Path

from hy3dshape import FaceReducer, FloaterRemover, DegenerateFaceRemover, Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.pipelines import export_to_trimesh
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.utils import logger
from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from hy3dpaint.convert_utils import create_glb_with_pbr_materials

HUNYUAN3D_REPO = "andreca/hunyuan3d-2.1xet"
HUNYUAN3D_DIT_MODEL = "hunyuan3d-dit-v2-1"
REALESRGAN_PATH = "/root/.cache/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"

class Output(BaseModel):
    mesh: Path

class Predictor(BasePredictor):
    def setup(self) -> None:
        try:
            start = time.time()
            logger.info("Setup started")
            os.environ["OMP_NUM_THREADS"] = "16"
            
            self.i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                HUNYUAN3D_REPO,
                subfolder=HUNYUAN3D_DIT_MODEL,
                use_safetensors=False,
                device="cuda"
            )
            
            conf = Hunyuan3DPaintConfig(max_num_view=8, resolution=768)
            conf.realesrgan_ckpt_path = REALESRGAN_PATH
            conf.multiview_pretrained_path = HUNYUAN3D_REPO
            conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
            conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
            self.texgen_worker = Hunyuan3DPaintPipeline(conf)
            
            self.floater_remove_worker = FloaterRemover()
            self.degenerate_face_remove_worker = DegenerateFaceRemover()
            self.face_reduce_worker = FaceReducer()
            self.rmbg_worker = BackgroundRemover()
            
            duration = time.time() - start
            logger.info(f"Setup took: {duration:.2f}s")
        except Exception as e:
            logger.error(f"Setup failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _cleanup_gpu_memory(self):
        if cuda.is_available():
            cuda.empty_cache()
            cuda.ipc_collect()

    def _log_analytics_event(self, event_name, params=None):
        pass

    def predict(
        self,
        image: Path = Input(
            description="Input image for generating 3D shape"
        ),
        steps: int = Input(
            description="Number of inference steps",
            default=50,
            ge=5,
            le=50,
        ),
        guidance_scale: float = Input(
            description="Guidance scale for generation",
            default=7.5,
            ge=1.0,
            le=20.0,
        ),
        max_facenum: int = Input(
            description="Maximum number of faces for mesh generation",
            default=20000,
            ge=10000,
            le=200000
        ),
        num_chunks: int = Input(
            description="Number of chunks for mesh generation",
            default=8000,
            ge=1000,
            le=200000
        ),
        seed: int = Input(
            description="Random seed for generation",
            default=1234
        ),
        octree_resolution: int = Input(
            description="Octree resolution for mesh generation",
            choices=[196, 256, 384, 512],
            default=256
        ),
        remove_background: bool = Input(
            description="Whether to remove background from input image",
            default=True
        ),
        generate_texture: bool = Input(
            description="Whether to generate PBR textures",
            default=True
        )
    ) -> Output:
        start_time = time.time()
        
        self._log_analytics_event("predict_started", {
            "steps": steps,
            "guidance_scale": guidance_scale,
            "max_facenum": max_facenum,
            "num_chunks": num_chunks,
            "seed": seed,
            "octree_resolution": octree_resolution,
            "remove_background": remove_background,
            "generate_texture": generate_texture
        })

        if os.path.exists("output"):
            shutil.rmtree("output")
        
        os.makedirs("output", exist_ok=True)

        self._cleanup_gpu_memory()

        generator = Generator()
        generator = generator.manual_seed(seed)

        if image is not None:
            input_image = Image.open(str(image))
            if remove_background or input_image.mode == "RGB":
                input_image = self.rmbg_worker(input_image.convert('RGB'))
                self._cleanup_gpu_memory()
        else:
            self._log_analytics_event("predict_error", {"error": "no_image_provided"})
            raise ValueError("Image must be provided")

        input_image.save("output/input.png")

        try:
            outputs = self.i23d_worker(
                image=input_image,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                octree_resolution=octree_resolution,
                num_chunks=num_chunks,
                output_type='mesh'
            )
            
            mesh = export_to_trimesh(outputs)[0]
            self._cleanup_gpu_memory()

            mesh = self.floater_remove_worker(mesh)
            mesh = self.degenerate_face_remove_worker(mesh)
            mesh = self.face_reduce_worker(mesh, max_facenum=max_facenum)
            self._cleanup_gpu_memory()
            
            if generate_texture:
                temp_mesh_path = "output/temp_mesh.obj"
                mesh.export(temp_mesh_path)
                
                textured_mesh_path = "output/textured_mesh.obj"
                self.texgen_worker(mesh_path=temp_mesh_path, image_path=input_image, 
                                 output_mesh_path=textured_mesh_path, save_glb=False)
                self._cleanup_gpu_memory()
                
                output_path = Path("output/textured_mesh.glb")
                textures = {
                    'albedo': textured_mesh_path.replace('.obj', '.jpg'),
                    'metallic': textured_mesh_path.replace('.obj', '_metallic.jpg'),
                    'roughness': textured_mesh_path.replace('.obj', '_roughness.jpg')
                }
                create_glb_with_pbr_materials(textured_mesh_path, textures, str(output_path))
            else:
                output_path = Path("output/mesh.glb")
                mesh.export(str(output_path), include_normals=True)

            if not Path(output_path).exists():
                self._log_analytics_event("predict_error", {"error": "mesh_export_failed"})
                raise RuntimeError(f"Failed to generate mesh file at {output_path}")

            duration = time.time() - start_time
            self._log_analytics_event("predict_completed", {
                "duration": duration,
                "final_face_count": len(mesh.faces),
                "success": True,
                "textured": generate_texture
            })

            return Output(mesh=output_path)
        except Exception as e:
            logger.error(f"Predict failed: {str(e)}")
            logger.error(traceback.format_exc())
            self._log_analytics_event("predict_error", {
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise 