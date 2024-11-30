import argparse
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description="Specify host and port for the server.")
    parser.add_argument('--host', type=str, default='127.0.0.1', help='The host address (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8000, help='The port number (default: 8080)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print debug info and save intermediate images in result folder')
    parser.add_argument('--start-instance', action='store_true',
                        help='If a translator should be launched automatically')
    parser.add_argument('--ignore-errors', action='store_true', help='Skip image on encountered error.')
    parser.add_argument('--nonce', default=os.getenv('MT_WEB_NONCE', ''), type=str, help='Nonce for securing internal web server communication')
    g = parser.add_mutually_exclusive_group()
    g.add_argument('--use-gpu', action='store_true', help='Turn on/off gpu (auto switch between mps and cuda)')
    g.add_argument('--use-gpu-limited', action='store_true', help='Turn on/off gpu (excluding offline translator)')
    return parser.parse_args()