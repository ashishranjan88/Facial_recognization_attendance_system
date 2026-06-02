import cv2

def open_cap(i, backend):
    if backend is None:
        cap = cv2.VideoCapture(i)
    else:
        cap = cv2.VideoCapture(i, backend)
    return cap


def try_backend(backend_name, backend):
    print(f"--- Backend: {backend_name} ---")
    for i in range(4):
        cap = open_cap(i, backend)
        opened = cap.isOpened()
        ret = False
        if opened:
            ret, _ = cap.read()
        print(f"Index {i}: isOpened={opened}, read_ok={ret}")
        if opened:
            cap.release()


if __name__ == '__main__':
    avfoundation = getattr(cv2, 'CAP_AVFOUNDATION', None)
    try_backend('default', None)
    if avfoundation is not None:
        try_backend('AVFoundation', avfoundation)
    else:
        print('AVFoundation backend not available in this OpenCV build.')
