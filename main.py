import cv2
import time
import os
import json
from dotenv import load_dotenv
from datetime import datetime
from pync import Notifier
import google.generativeai as genai

load_dotenv()

# Gemini モデルを設定
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
model = genai.GenerativeModel('gemini-1.5-flash')


def upload_image(image_path):
    return genai.upload_file(path=image_path)


def analyze_image(file):
    prompt = """
    画像に写っている人物の表情を分析し、以下の形式のJSONで回答してください：
    
    - "sleepy"は人物が眠そうに見える場合はtrue、そうでない場合はfalseにしてください。
    - "troubled"は人物が困っているように見える場合はtrue、そうでない場合はfalseにしてください。
    - 必ず"{"で始まり"}"で終わる形式でレスポンスを返してください。
    
    レスポンスの形式:
    {
        "sleepy": true/false,
        "troubled": true/false
    }
    """

    try:
        response = model.generate_content([prompt, file])

        print("API Response:", response.text)

        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            sleepy = "true" in response.text.lower() and "sleepy" in response.text.lower()
            troubled = "true" in response.text.lower() and "troubled" in response.text.lower()
            return {"sleepy": sleepy, "troubled": troubled}

    except Exception as e:
        print(f"API呼び出し中にエラーが発生しました: {str(e)}")
        return {"sleepy": False, "troubled": False}


def send_notification(message, title):
    Notifier.notify(
        message,
        title=title,
        sound='default'
    )


def capture_images():
    cap = cv2.VideoCapture(0)
    save_dir = "captured_images"
    os.makedirs(save_dir, exist_ok=True)

    try:
        while True:
            ret, frame = cap.read()

            if ret:
                # 既存の画像ファイルを取得し、タイムスタンプでソート
                existing_files = sorted([f for f in os.listdir(save_dir) if f.endswith('.jpg')])

                # ローカルのファイル数が5つ以上の場合、古いファイルを削除
                while len(existing_files) >= 5:
                    oldest_file = existing_files.pop(0)
                    os.remove(os.path.join(save_dir, oldest_file))
                    print(f"古い画像を削除しました: {oldest_file}")

                # 新しい画像を保存
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{save_dir}/image_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"画像を保存しました: {filename}")

                # 画像をアップロード
                uploaded_file = upload_image(filename)
                print(f"画像をアップロードしました: {uploaded_file.uri}")

                # Gemini APIで画像を分析
                analysis_result = analyze_image(uploaded_file)
                print("分析結果:", analysis_result)

                # 分析結果に基づいて通知を送信
                if analysis_result.get('troubled', False):
                    send_notification('何か困っていますか？', '状態確認')

                if analysis_result.get('sleepy', False):
                    send_notification('眠そうですね。少しリフレッシュしてみましょう', '状態確認')

                # 5秒待機
                time.sleep(5)
            else:
                print("フレームの取得に失敗しました")
                break

    except KeyboardInterrupt:
        print("プログラムを終了します")

    finally:
        cap.release()


if __name__ == "__main__":
    capture_images()
