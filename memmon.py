import sys
import time

import psutil

def memMon(pid, freq=1.):
    proc = psutil.Process(pid)
    print(proc.memory_info())
    prev_mem = None
    while True:
        try:
            mem = proc.memory_info().rss / 1e6
            if prev_mem is None:
                print('{:10.3f} [Mb]'.format(mem))
            elif abs(mem - prev_mem) > 0:
                print('{:10.3f} [Mb] {:+10.3f} [Mb]'.format(mem, mem - prev_mem))
            prev_mem = mem
            time.sleep(freq)
        except KeyboardInterrupt:
            try:
                input(' Pausing memMon, <cr> to continue, ^C to end...')
            except KeyboardInterrupt:
                print('\n')
                return

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python pidmon.py <PID>')
        sys.exit(1)
    pid = int(sys.argv[1])
    memMon(pid)
    sys.exit(0)
