import http.server
import socketserver
from http import HTTPStatus
import os
import subprocess
import time

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if(self.path.startswith("/reset")):
            os.system("cd .. && docker-compose restart")
            for _ in range(15):
                process = subprocess.Popen(r'''curl -s -o /dev/null -w "%{http_code}" http://localhost/''', shell=True, stdout=subprocess.PIPE)
                stdout = process.communicate()[0]
                print(stdout)
                if int(stdout) == 200:
                    self.send_response(HTTPStatus.OK)
                    self.end_headers()
                    self.wfile.write(b'Reset successfully')
                    break
                time.sleep(1)
            else:
                self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR)
                self.end_headers()
                self.wfile.write(b'Resetting container timeout')            

        else:
            self.send_response(HTTPStatus.OK)
            self.end_headers()
            self.wfile.write(b'Hello world')
        
def main():
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(('', 47897), Handler)
    httpd.serve_forever()

if __name__ == "__main__":
    main()