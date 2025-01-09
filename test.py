import cv2

def main(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    frame_number = 0
    frames = []

    while cap.isOpened():
        if frame_number == frame_index:
            ret, frame = cap.read()
            frames.append(frame)
            if len(frames)>=200:
                frames[len(frames)-200] = None
            frame_number += 1
        frame = frames[frame_index]
        cv2.imshow('Video', frame)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            file_full_name = f"{save_path}\\frame_{frame_index}.jpg"
            res = cv2.imwrite(file_full_name, frame)
            print(res, "图片保存完成 ", file_full_name)
        elif key == ord('d'):  # Left arrow key
            frame_index += 1
        elif key == ord('a') and frame_index > frame_number-200 and frame_index > 0:  # Right arrow key
            frame_index -= 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = r'C:\Users\24225\Desktop\2025-01-03\10.70.79.200_01_20250103101736342.mp4'
    save_path = 'C:\\Users\\24225\\Desktop\\2025-01-03'
    main(video_path, save_path)