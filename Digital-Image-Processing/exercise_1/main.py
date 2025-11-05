import sys
from dip1 import run

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_image>")
        return -1

    fname = sys.argv[1]
    try:
       run(fname)
    except Exception as e:
        print("An error occurred:")
        print(e)
        return -1
    run(fname)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
