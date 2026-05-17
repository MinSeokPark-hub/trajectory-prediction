import glob
from src.utils.kitti_parser import KittiParser
from src.utils.sgan_parser import SGANParser

# KITTI 테스트
p = KittiParser()
df = p.load("/workspace/minseok_park/data/kitti/labels/0000.txt")
print("KITTI 컬럼:", df.columns.tolist())
print("KITTI shape:", df.shape)
print()

# SGAN 테스트
sgan_files = glob.glob("/workspace/minseok_park/data/sgan/datasets/eth/train/*.txt")
p2 = SGANParser()
df2 = p2.load(sgan_files[0])
print("SGAN 컬럼:", df2.columns.tolist())
print("SGAN shape:", df2.shape)