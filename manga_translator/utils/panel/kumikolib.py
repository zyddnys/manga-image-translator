import os
import sys
import tempfile
import cv2 as cv
import numpy as np
import requests
import subprocess
from urllib.parse import urlparse

from .lib.page import Page, NotAnImageException
from .lib.debug import Debug


class Kumiko:

	options = {}

	def __init__(self, options = None):
		options = options or {}

		for o in ['progress', 'rtl', 'debug']:
			self.options[o] = o in options and options[o]

		Debug.debug = self.options['debug']

		self.options['min_panel_size_ratio'] = options.get('min_panel_size_ratio', None)

		self.panel_expansion = options.get('panel_expansion', True)

		self.page_list = []

	def parse_url_list(self, urls):
		if self.options['progress']:
			print(len(urls), 'files to download', file = sys.stderr)

		self.temp_folder = tempfile.TemporaryDirectory()

		i = 0
		nbdigits = len(str(len(urls)))
		for url in urls:
			filename = 'img' + ('0' * nbdigits + str(i))[-nbdigits:]

			if self.options['progress']:
				print('\t', url, (' -> ' + filename) if urls else '', file = sys.stderr)

			i += 1
			parts = urlparse(url)
			if not parts.netloc or not parts.path:
				continue

			r = requests.get(url, timeout = 5)
			with open(os.path.join(self.temp_folder.name, filename), 'wb') as f:
				f.write(r.content)

		self.parse_dir(self.temp_folder.name, urls = urls)

	def parse_pdf_file(self, pdf_filename):
		try:
			subprocess.run(args = ['pdftoppm', '--help'], check = True, capture_output = True)
		except FileNotFoundError:
			print("Please `apt install pdftoppm` if you give PDF --input files to Kumiko", file = sys.stderr)
			sys.exit(1)

		self.temp_folder = tempfile.mkdtemp(prefix = "kumiko-pdf-pages-")

		print(f"Using pdftoppm to extract jpeg files from pdf to {self.temp_folder}", file = sys.stderr)
		subprocess.run(args = ['pdftoppm', '-jpeg', pdf_filename, f"{self.temp_folder}/"], check = True)

		self.parse_dir(self.temp_folder)

	def parse_dir(self, directory, urls = None):
		filenames = []
		for filename in os.scandir(directory):
			filenames.append(filename.path)
		self.parse_images(filenames, urls)

	def parse_images(self, filenames, urls = None):
		if self.options['progress']:
			print(len(filenames), 'files to cut panels for', file = sys.stderr)

		i = -1
		for filename in sorted(filenames):
			i += 1
			if self.options['progress']:
				print("\t", urls[i] if urls else filename, file = sys.stderr)

			try:
				self.parse_image(filename, url = urls[i] if urls else None)
			except NotAnImageException:
				if not filename.endswith(".license"):
					print(f"\n[ERROR] Not an image, will be ignored: {filename}\n", file = sys.stderr)

	def parse_image(self, filename, url = None):
		self.page_list.append(
			Page(
				filename,
				numbering = "rtl" if self.options['rtl'] else "ltr",
				url = url,
				min_panel_size_ratio = self.options['min_panel_size_ratio'],
				panel_expansion = self.panel_expansion,
			)
		)

	def get_infos(self):
		return list(map(lambda p: p.get_infos(), self.page_list))

	def save_panels(self, output_base_path = 'auto', output_format = "jpg"):
		if output_base_path == 'auto':
			output_base_path = tempfile.mkdtemp(prefix = "kumiko-out-")
		elif not os.path.isdir(output_base_path):
			print(
				f"\n[ERROR] Given --save-panels directory is not a directory: {output_base_path}\n", file = sys.stderr
			)
			sys.exit(1)

		nb_written_panels = 0
		for page in self.page_list:
			output_path = os.path.join(output_base_path, os.path.basename(page.filename))
			os.makedirs(output_path, exist_ok = True)

			for i, panel in enumerate(page.panels):
				x, y, width, height = panel.to_xywh()
				output_file = os.path.join(output_path, f"panel_{i}.{output_format}")
				panel = page.img[y:y + height, x:x + width]
				if cv.imwrite(output_file, panel):
					nb_written_panels += 1
				else:
					print(f"\n[ERROR] Failed to write panel image to {output_file}\n", file = sys.stderr)

		print(f"Saved {nb_written_panels} panel images to {output_base_path}", file = sys.stderr)
