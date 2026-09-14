import sys
import re
import yt_dlp
import threading
import datetime
from InquirerPy import inquirer

LOG_FILE = "download.log"

def log_message(message):
    """Logs a message to the log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {message}\n")

def to_camel_case(name):
    """Normalizes string to CamelCase."""
    name = re.sub(r'[^a-zA-Z0-9 ]', '', name)
    words = name.split()
    return "".join(word.capitalize() for word in words)

def search_youtube(query, max_results=5):
    """Searches YouTube and returns a list of dictionaries with title and url."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
        'extract_flat': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            results = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
            if 'entries' in results:
                return [{'title': entry['title'], 'url': entry['url']} for entry in results['entries']]
            else:
                return []
        except Exception as e:
            log_message(f"Error searching YouTube for '{query}': {e}")
            return []

def download_audio_task(video_url, title):
    """Downloads audio for a given URL in the background."""
    clean_title = to_camel_case(title)
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': f'{clean_title}.%(ext)s',
        'logger': None, # Suppress noisy output to console
        'quiet': True,
    }

    log_message(f"Starting download: {clean_title}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([video_url])
            log_message(f"Successfully downloaded: {clean_title}.mp3")
        except Exception as e:
            log_message(f"Error downloading {clean_title}: {e}")

def main():
    print("Welcome to the Async YouTube Audio Downloader!")
    print(f"Downloads will be logged to {LOG_FILE}.")

    while True:
        query = input("\nEnter search term (or 'q' to quit): ")
        if query.lower() == 'q':
            break

        print(f"Searching for '{query}'...")
        videos = search_youtube(query)

        if not videos:
            print("No videos found.")
            continue

        selected_titles = inquirer.checkbox(
            message="Select videos to download:",
            choices=[v['title'] for v in videos],
            cycle=True
        ).execute()

        for title in selected_titles:
            video = next(v for v in videos if v['title'] == title)
            thread = threading.Thread(target=download_audio_task, args=(video['url'], title))
            thread.daemon = True
            thread.start()
            print(f"Queued for download: {title}")

if __name__ == "__main__":
    main()
