import os
import shutil
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

HUNYUAN3D_REPO = "tencent/Hunyuan3D-2.1"
HUNYUAN3D_DIT_MODEL = "hunyuan3d-dit-v2-1"
REALESRGAN_PATH = "/root/.cache/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"

def export_mesh(mesh, save_folder, textured=False, type='glb'):
    """
    Export a mesh to a file in the specified folder, optionally including textures.

    Args:
        mesh (trimesh.Trimesh): The mesh object to export.
        save_folder (str): Directory path where the mesh file will be saved.
        textured (bool, optional): Whether to include textures/normals in the export. Defaults to False.
        type (str, optional): File format to export ('glb' or 'obj' supported). Defaults to 'glb'.

    Returns:
        str: The full path to the exported mesh file.
    """
    if textured:
        path = os.path.join(save_folder, f'textured_mesh.{type}')
    else:
        path = os.path.join(save_folder, f'white_mesh.{type}')
    if type not in ['glb', 'obj']:
        mesh.export(path)
    else:
        mesh.export(path, include_normals=textured)
    return path

def quick_convert_with_obj2gltf(obj_path: str, glb_path: str) -> bool:
    """Convert textured OBJ to GLB with PBR materials."""
    try:
        textures = {
            'albedo': obj_path.replace('.obj', '.jpg'),
            'metallic': obj_path.replace('.obj', '_metallic.jpg'),
            'roughness': obj_path.replace('.obj', '_roughness.jpg')
        }
        create_glb_with_pbr_materials(obj_path, textures, glb_path)
        return True
    except Exception as e:
        logger.error(f"Failed to convert OBJ to GLB: {e}")
        return False

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
            conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
            conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
            self.tex_pipeline = Hunyuan3DPaintPipeline(conf)
            
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
            description="Input image for generating 3D shape",
            default=None
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
            # Generate shape using the new API
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
                temp_mesh_path = export_mesh(mesh, "output", textured=False, type='obj')
                
                textured_mesh_path = os.path.join("output", "textured_mesh.obj")
                self.tex_pipeline(mesh_path=temp_mesh_path, image_path=input_image, 
                                output_mesh_path=textured_mesh_path, save_glb=False)
                self._cleanup_gpu_memory()
                
                output_path = Path("output/textured_mesh.glb")
                conversion_success = quick_convert_with_obj2gltf(textured_mesh_path, str(output_path))
                
                if not conversion_success:
                    logger.warning("GLB conversion failed, falling back to OBJ")
                    output_path = Path(textured_mesh_path)
            else:
                output_path = Path(export_mesh(mesh, "output", textured=False, type='glb'))

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