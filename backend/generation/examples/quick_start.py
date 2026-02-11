"""
快速开始示例
多模态生成功能使用示例
"""

import asyncio
from generation.unified import get_generation_manager


async def main():
    """主函数示例"""
    manager = get_generation_manager()
    
    # ============ 图像生成示例 ============
    print("=== 图像生成示例 ===")
    image_response = await manager.generate(
        modality="image",
        prompt="一只可爱的橘猫坐在窗台上，看着外面的雨天",
        provider="dalle",
        size="1024x1024",
        num_images=1
    )
    
    if image_response.success:
        print(f"✅ 图像生成成功! Task ID: {image_response.task_id}")
        print(f"   Provider: {image_response.provider}")
        print(f"   Model: {image_response.model}")
    else:
        print(f"❌ 图像生成失败: {image_response.error}")
    
    # ============ 音频生成示例 ============
    print("\n=== 音频生成示例 ===")
    audio_response = await manager.generate(
        modality="audio",
        text="你好！欢迎使用AI Platform多模态生成服务。",
        provider="openai",
        voice="nova",
        speed=1.0,
        format="mp3"
    )
    
    if audio_response.success:
        print(f"✅ 音频生成成功! Task ID: {audio_response.task_id}")
        print(f"   Duration: {audio_response.duration_ms}ms")
        print(f"   Characters: {audio_response.characters}")
    else:
        print(f"❌ 音频生成失败: {audio_response.error}")
    
    # ============ 视频生成示例 ============
    print("\n=== 视频生成示例 ===")
    video_response = await manager.generate(
        modality="video",
        prompt="一座古老的城堡在山脚下，夕阳西下，金色的阳光洒在城墙上",
        provider="sora",
        duration="5-10秒",
        resolution="1080x1920",
        fps=24
    )
    
    if video_response.success:
        print(f"✅ 视频生成成功! Task ID: {video_response.task_id}")
        print(f"   Duration: {video_response.duration_ms}ms")
        print(f"   Resolution: {video_response.resolution}")
    else:
        print(f"❌ 视频生成失败: {video_response.error}")
    
    # ============ 多模态生成示例 ============
    print("\n=== 多模态生成示例 ===")
    multimodal_results = await manager.generate_multimodal(
        prompt="一只小狗在草地上奔跑",
        modalities=["image", "audio"],
        providers={
            "image": "dalle",
            "audio": "openai"
        }
    )
    
    for modality, response in multimodal_results.items():
        if response.success:
            print(f"✅ {modality} 生成成功!")
        else:
            print(f"❌ {modality} 生成失败: {response.error}")
    
    # ============ 获取可用模型 ============
    print("\n=== 可用模型列表 ===")
    models = await manager.get_available_models()
    for modality, model_list in models.items():
        print(f"\n{modality.upper()}:")
        for model in model_list:
            print(f"   - {model['id']} ({model['provider']})")
    
    # ============ 成本估算 ============
    print("\n=== 成本估算示例 ===")
    cost = await manager.estimate_cost(
        modality="image",
        provider="dalle",
        params={"num_images": 1}
    )
    if cost.get("estimated"):
        print(f"   预估成本: ${cost['total_cost']} USD")
    else:
        print(f"   成本估算失败: {cost.get('message')}")


if __name__ == "__main__":
    asyncio.run(main())
