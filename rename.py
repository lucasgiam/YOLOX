import os
from pathlib import Path

# imgdir = "./datasets/sp_ppe_2/VOCdevkit/VOC2012/JPEGImages"
imgdir = "C:/Users/Lucas_Giam/Desktop/datasets_2/VOCdevkit/VOC2012/JPEGImages"
# xmldir = "./datasets/sp_ppe_2/VOCdevkit/VOC2012/Annotations"
xmldir = "C:/Users/Lucas_Giam/Desktop/datasets_2/VOCdevkit/VOC2012/Annotations"

def main():
    imgnames_orig = sorted(list(Path.iterdir(Path(imgdir))))
    for i, imgname in enumerate(imgnames_orig):
        renamerobo(imgname)
        xmlname = Path(xmldir) / imgname.with_suffix(".xml").name
        renamerobo(xmlname)
        # xmlname = Path(xmldir) / imgname.with_suffix(".xml").name
        # appendimg(xmlname)

# def appendimg(src):
#     src1 = src.stem
#     src2, src3 = src1.split(" ")
#     src4 = src2 + "_" + src3[1:-1]
#     os.rename(src, src.with_stem(src4))
#     print(src)
#     print(src.with_stem(src4))

def renamerobo(src):
    src1 = src.stem
    src2, src3 = src1.split("--")
    src4 = src3.split("-")[0]
    src5 = src2 + "_" + src4
    os.rename(src, src.with_stem(src5))
    print(src)
    print(src.with_stem(src5))


if __name__ == "__main__":
    main()
