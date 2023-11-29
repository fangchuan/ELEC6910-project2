import cv2

def loadVid(path):
    # Create a VideoCapture object to read the video
    cap = cv2.VideoCapture(path)

    # Get video properties
    nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vidHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vidWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Preallocate movie structure
    mov = [None] * nFrames

    # Read one frame at a time
    for k in range(nFrames):
        print(f'{k + 1}/{nFrames}')
        ret, frame = cap.read()
        if ret:
            mov[k] = {'cdata': frame, 'colormap': None}

    # Release the VideoCapture object
    cap.release()

    return mov