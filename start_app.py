# Author: wujiahang
import os, sys

BASE = os.path.dirname(__file__)
if BASE not in sys.path: sys.path.insert(0, BASE)
from pose_coach.main import main

if __name__ == "__main__":
    main()