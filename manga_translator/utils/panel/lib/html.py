import json


class HTML:

	@staticmethod
	def header(title = '', reldir = ''):
		return f"""<!DOCTYPE html>
			<html>

			<head>
				<title>Kumiko Reader</title>
				<meta charset="utf-8">
				<meta name="viewport" content="width=device-width, initial-scale=1">
				<script type="text/javascript" src="{reldir}jquery-3.2.1.min.js"></script>
				<script type="text/javascript" src="{reldir}reader.js"></script>
				<link rel="stylesheet" media="all" href="{reldir}style.css" />
				<style type="text/css">
					h2, h3 {{ text-align: center; margin-top: 3em; }}
					.sidebyside {{ display: flex; justify-content: space-around; }}
					.sidebyside > div {{ width: 45%; }}
					.version, .step-info {{ text-align: center; }}
					.kumiko-reader.halfwidth {{ max-width: 45%; max-height: 90vh; }}
					.kumiko-reader.fullpage {{ width: 100%; height: 100%; }}
				</style>
			</head>

			<body>
			<h1>{title}</h1>

		"""

	@staticmethod
	def nbdiffs(files_diff):
		return f"<p>{len(files_diff)} differences found in files</p>"

	pageId = 0

	@staticmethod
	def side_by_side_panels(title, step_info, jsons, v1, v2, images_dir, known_panels, diff_numbering_panels):
		html = f"""
			<h2>{title}</h2>
			<p class="step-info">{step_info}</p>
			<div class="sidebyside">
				<div class="version">{v1} <span class="processing_time">− processing time {jsons[0][0]['processing_time'] if 'processing_time' in jsons[0][0] else "??"}s</span></div>
				<div class="version">{v2} <span class="processing_time">- processing time {jsons[1][0]['processing_time']}s</span></div>
			</div>
			<div class="sidebyside">
		"""

		oneside = """
			<div id="page{id}" class="kumiko-reader halfwidth debug"></div>
			<script type="text/javascript">
				var reader = new Reader({{
					container: $('#page{id}'),
					comicsJson: {json},
					images_dir: {images_dir},
					known_panels: {known_panels},
					diff_numbering_panels: {diff_numbering_panels},
				}});
				reader.start();
			</script>
			"""
		i = -1
		for js in jsons:
			i += 1
			html += oneside.format(
				id = HTML.pageId,
				json = json.dumps(js),
				images_dir = json.dumps(images_dir),
				known_panels = known_panels[i],
				diff_numbering_panels = diff_numbering_panels
			)
			HTML.pageId += 1

		html += '</div>'
		return html

	@staticmethod
	def imgbox(images):
		html = "<h3>Debugging images</h3>\n<div class='imgbox'>\n"
		for img in images:
			html += f"\t<div><p>{img['label']}</p><img src='{img['filename']}' /></div>\n"

		return html + "</div>\n\n"

	@staticmethod
	def reader(js, images_dir):
		return f"""
			<div id="reader" class="kumiko-reader fullpage"></div>
			<script type="text/javascript">
				var reader = new Reader({{
					container: $('#reader'),
					comicsJson: {js},
					images_dir: {json.dumps(images_dir)},
					controls: true
				}});
				reader.start();
			</script>
			"""

	footer = """

</body>
</html>
"""
