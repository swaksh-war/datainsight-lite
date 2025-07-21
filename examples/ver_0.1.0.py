import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
from datainsight_lite.datainsight import quick_report

quick_report('test_with_categorical.csv')