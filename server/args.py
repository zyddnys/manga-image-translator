import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Specify host and port for the server.")
    parser.add_argument('--host', type=str, default='127.0.0.1', help='The host address (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8000, help='The port number (default: 8080)')
    #use_gpu
    #use_gpu_limited
    #ignore_errors
    #verbose
    #nonce
    #start_instance
    return parser.parse_args()