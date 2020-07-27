import socket
import base64
from json import dumps
HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 6135        # The port used by the server

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        with open("ben.jpeg", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            #print(len(encoded_string))

        s.connect((HOST, PORT))

        s.sendall('0'.encode())
        data = s.recv(1024)

        if data.decode() == "ACK":
            s.sendall(encoded_string+"bay".encode())
            data = s.recv(1024)
            print(data)

        d = {'name': "yahel",
             'desc': "manor",
             'addr': "home",
             'gender': "male",
             }
        print(1)
        print(s.recv(1024).decode())

        s.sendall("bay".encode())

        data = s.recv(1024)
        s.sendall('1'.encode())
        for x in range(5):
            if data.decode() == "ACK":
                s.sendall(encoded_string + "bay".encode())
                data = s.recv(1024)
                print(data)
        data = s.recv(1024)
        print(dumps(d))
        s.sendall(dumps(d).encode())
        s.recv(10000)
    print(data.decode())


main()