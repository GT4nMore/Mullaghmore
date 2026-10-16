"""Dev server: serves ../docs and saves screenshots POSTed by the viewer (?shot=name).

    python shotserver.py            # serve on 127.0.0.1:8765
    python shotserver.py take NAME "x,y,z,yaw,pitch" [WxH]
"""
import base64
import os
import subprocess
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.environ.get('SHOT_SITE', os.path.join(HERE, '..', 'docs'))
WORK = os.path.join(HERE, '..', 'work')
CHROME = r'C:/Program Files/Google/Chrome/Application/chrome.exe'


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=SITE, **kw)

    def do_POST(self):
        name = self.path.split('name=')[-1]
        data = self.rfile.read(int(self.headers['Content-Length']))
        if self.path.startswith('/probe'):
            with open(os.path.join(WORK, 'probe.txt'), 'wb') as f:
                f.write(data)
            self.send_response(200)
            self.end_headers()
            return
        png = base64.b64decode(data.split(b',', 1)[1])
        with open(os.path.join(WORK, name + '.png'), 'wb') as f:
            f.write(png)
        self.send_response(200)
        self.end_headers()

    def log_message(self, *a):
        pass


def take(name, cam, size='1280x720', timeout=180):
    out = os.path.join(WORK, name + '.png')
    if os.path.exists(out):
        os.remove(out)
    w, h = size.split('x')
    prof = os.path.join(WORK, 'prof-' + name)
    p = subprocess.Popen([CHROME, '--headless=new', '--use-gl=angle', '--use-angle=swiftshader',
                          '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', f'--window-size={w},{h}',
                          '--remote-debugging-port=0', f'--user-data-dir={prof}',
                          f'http://127.0.0.1:8765/index.html?shot={name}&cam={cam}'],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t = time.time()
    while not os.path.exists(out) and time.time() - t < timeout:
        time.sleep(0.5)
    time.sleep(0.5)
    p.kill()
    print(name, 'ok' if os.path.exists(out) else 'TIMEOUT')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'take':
        take(*sys.argv[2:])
    else:
        ThreadingHTTPServer(('127.0.0.1', 8765), Handler).serve_forever()
