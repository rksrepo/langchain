import os

from dotenv import load_dotenv

load_dotenv(override=True)


def main():
    print("Hello world")
    print("OpenAI API Key:: ", os.getenv("OPENAI_API_KEY"))


if __name__ == "__main__":
    main()
