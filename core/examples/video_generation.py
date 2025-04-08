import asyncio

from core.components import LLMProvider, MediaHandler


async def main():
    llm_provider = LLMProvider()
    media_handler = MediaHandler(llm_provider)
    user_message = "video of a cat playing with a ball"
    video_prompt = await media_handler.generate_video_prompt(user_message, duration=10)
    print(f"Generated video prompt: {video_prompt}")
    video_url = await media_handler.generate_video(prompt=video_prompt, duration=10)
    print(f"Video URL: {video_url}")


if __name__ == "__main__":
    asyncio.run(main())
