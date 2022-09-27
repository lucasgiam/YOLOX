import hashlib
from pathlib import Path

full_path = "./datasets/sp_ppe_2/VOCdevkit/VOC2012/JPEGImages"
# full_path = "C:/Users/Lucas_Giam/Desktop/datasets_2/VOCdevkit/VOC2012/JPEGImages"

def main():
    imgnames_orig = Path.iterdir(Path(full_path))
    for i, imgname in enumerate(imgnames_orig):
        print(imgname, hashlib.md5(open(imgname, 'rb').read()).hexdigest())
        break

if __name__ == "__main__":
    main()
