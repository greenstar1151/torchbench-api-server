import http.server
import socketserver
from http import HTTPStatus
import os

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if(self.path.startswith("/reset")):
            os.system("cd .. && docker-compose restart")
            self.send_response(HTTPStatus.OK)
            self.end_headers()
            self.wfile.write(b'Reset successfully')
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