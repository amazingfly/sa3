from http.server import HTTPServer, SimpleHTTPRequestHandler
print("Starting...")
httpd = HTTPServer(("0.0.0.0", 7861), SimpleHTTPRequestHandler)
print("Started.")
# httpd.serve_forever() # Don't block
