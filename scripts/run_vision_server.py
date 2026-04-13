from vision_server import VisionServer
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Vision Server with optional SAM3 Masking.")
    parser.add_argument("--use-sam3", action="store_true", help="Enable SAM3 masking on the camera streams.")
    args = parser.parse_args()

    # HuggingFace 토큰 설정 (SAM3 가중치 다운로드 용도)
    hf_token = <YOUR_TOKEN>
    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    MY_CAMERAS = {
        "wrist": "844212070094",
        "left":  "16FB99DF",
        "right": "2F1C99DF"
    }
    
    PROMPTS = {
        "wrist": "wristwatch",
        "left":  "wristwatch",
        "right": "wristwatch"
    }

    server = VisionServer(
        camera_configs=MY_CAMERAS, 
        port=5555, 
        use_sam3=args.use_sam3,
        text_prompts=PROMPTS
    )
    server.start()