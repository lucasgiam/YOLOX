import os
from pathlib import Path

imgdir = r'C:\Users\Lucas_Giam\Desktop\c+\img'
xmldir = r'C:\Users\Lucas_Giam\Desktop\c+\xml'

def main():
    imgnames = sorted(list(Path.iterdir(Path(imgdir))))
    for i, imgname in enumerate(imgnames):
        append(imgname)
        xmlname = Path(xmldir) / imgname.with_suffix(".xml").name
        append(xmlname)

def append(src):
    src1 = src.stem
    src2, src3 = src1.split("_")
    src4 = src2.replace("+","_")
    src5 = src4 + "_" + src3
    os.rename(src, src.with_stem(src5))
    # print(src)
    # print(src.with_stem(src5))

# def renamerobo(src):
#     src1 = src.stem
#     src2, src3 = src1.split("--")
#     src4 = src3.split("-")[0]
#     src5 = src2 + "_" + src4
#     os.rename(src, src.with_stem(src5))
#     print(src)
#     print(src.with_stem(src5))


if __name__ == "__main__":
    main()
