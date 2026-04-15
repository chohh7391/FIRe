from vision_server import VisionServer
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Vision Server with optional SAM3 Masking.")
    parser.add_argument("--use_sam3", action="store_true", help="Enable SAM3 masking on the camera streams.")
    parser.add_argument("--target_object", type=str, default="peg")
    args = parser.parse_args()

    # HuggingFace 토큰 설정 (SAM3 가중치 다운로드 용도)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    MY_CAMERAS = {
        "wrist": "844212070094",
        "left":  "16FB99DF",
        "right": "2F1C99DF"
    }

    target_object = args.target_object
    
    PROMPTS = {
        "wrist": target_object,
        "left":  target_object,
        "right": target_object
    }

    server = VisionServer(
        camera_configs=MY_CAMERAS, 
        port=5555, 
        use_sam3=args.use_sam3,
        text_prompts=PROMPTS
    )
    server.start()