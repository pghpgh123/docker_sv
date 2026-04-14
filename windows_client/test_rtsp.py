import subprocess
import argparse

def push_audio_to_rtsp(input_wav, rtsp_url, sample_rate=16000, channels=1):
    cmd = [
        'ffmpeg', '-re', '-i', input_wav,
        '-vn', '-acodec', 'pcm_s16le', '-ar', str(sample_rate), '-ac', str(channels),
        '-f', 'rtsp', rtsp_url
    ]
    print(' '.join(cmd))
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description='RTSP音频推流测试脚本')
    parser.add_argument('--input_wav', required=True, help='本地音频文件路径')
    parser.add_argument('--rtsp_url', required=True, help='目标RTSP地址')
    args = parser.parse_args()
    push_audio_to_rtsp(args.input_wav, args.rtsp_url)

if __name__ == '__main__':
    main()
