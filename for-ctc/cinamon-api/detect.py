import sys
import time
from lib import detect_all
from pprint import pprint

def main():
    path = sys.argv[1]
    start_time = time.time()
    result = detect_all(path)
    stop_time = time.time()
    pprint(result)
    print(f'Took {stop_time - start_time}')

if __name__=='__main__':
    main()

